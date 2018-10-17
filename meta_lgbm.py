import pandas as pd
import numpy as np
import os, time, tqdm
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Import train and test meta data
train = pd.read_csv(os.getcwd() + '/' + 'data/training_set_metadata.csv')
test = pd.read_csv(os.getcwd() + '/' + 'data/test_set_metadata.csv')

# Check overall info
print(train.columns, train.shape)
print(test.columns, test.shape)

# Check for NAs in metadata
print('Num. of NAs:')
print(train.isna().sum())
print(test.isna().sum())

# Remove spectrometry redshift, distmod feats from both sets
feats_to_delete = ['hostgal_specz', 'distmod']
train.drop(feats_to_delete, axis=1, inplace=True)
test.drop(feats_to_delete, axis=1, inplace=True)

# Feat eng.
for df in [train, test]:
    df['is_galactic'] = df['hostgal_photoz'] == 0

# Check feat was created
print(train.columns, train.shape)
print(test.columns, test.shape)

# Check class freqs.
print(train.groupby('target')['object_id'].count())

# Get targets
y_tgt = train['target'].values
train.drop(['target'], axis=1)

# Get feat col names
train_cols = list(train.columns)
[train_cols.remove(c) for c in ['object_id', 'target']]

# Setup stratified CV
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Get sorted class weights
# class_weights_dict = {
#     #99 : 2.002408,
#     95 : 1.001044,
#     92 : 1.001044,
#     90 : 1.001044,
#     88 : 1.001044,
#     67 : 1.001044,
#     65 : 1.001044,
#     64 : 2.007104,
#     62 : 1.001044,
#     53 : 1.000000,
#     52 : 1.001044,
#     42 : 1.001044,
#     16 : 1.001044,
#     15 : 2.001886,
#     6 : 1.001044,
# }
class_weights_dict = {
    #99 : 2.002408,
    95 : 1,
    92 : 1,
    90 : 1,
    88 : 1,
    67 : 1,
    65 : 1,
    64 : 1,
    62 : 1,
    53 : 1,
    52 : 1,
    42 : 1,
    16 : 1,
    15 : 1,
    6 : 1,
}
class_codes = np.unique(list(class_weights_dict.keys()))
class_weights = {i : class_weights_dict[c] for i,c in enumerate(class_codes)}
label_encode = {c: i for i, c in enumerate(class_codes)}
weights = np.array([class_weights[i] for i in range(len(class_weights))])

def lgbm_loss_wrapper(y_true, y_pred):
    '''
    Custom lgbm eval metric - multiclass weighted logloss
    :param y_true: Encoded 1D targets, needs OHE
    :param y_pred: 1D preds class-first according to lgbm doc
    :return: 
    '''

    # Reshape preds from 1D lgb class-first format to standard 2D format
    num_classes = np.max(y_true) + 1
    y_pred = np.reshape(y_pred, (y_true.shape[0], num_classes), order='F')

    return ('lgbm_custom_logloss', weighted_mc_crossentropy(y_true, y_pred, weighted=True), False)

def loss_wrapper(y_true, y_pred):
    '''
    Custom eval metric, same as above only inputs are in different format
    :param y_true: Unencoded targets, 1D, needs OHE
    :param y_pred: 2D array of probabilities, summing to 1 per row (across classes)
    :return: 
    '''

    # Encode targets
    y_true = np.array([label_encode[tgt] for tgt in y_true])

    return ('custom_logloss', weighted_mc_crossentropy(y_true, y_pred, weighted=True), False)

def weighted_mc_crossentropy(y_true, y_pred, weighted=True):
    '''
    Custom defined weight multiclass cross entropy
    :param y_true: Encoded 1D targets, needs OHE
    :param y_pred: 2D array of probabilities, summing to 1 per row (across classes)
    :param weighted: bool - If True all weights will be set to 1
    :return: (float) Calculated loss
    '''

    num_classes = np.max(y_true) + 1

    # OHE target
    y_true = np.eye(num_classes)[y_true]

    # Clip preds for log safety
    y_pred = np.log(np.clip(y_pred, 1e-15, 1 - 1e-15))

    if weighted:
        # Compute avg. loss per class
        avg_loss_per_class = np.sum(y_true * y_pred, axis=0) / np.sum(y_true, axis=0)
        # Weighted avg. of class losses
        return - np.sum(weights * avg_loss_per_class) / np.sum(weights)

    else:
        return - np.sum(y_true * y_pred) / y_true.shape[0]

def save_importances(imps_):
    mean_gain = imps_[['gain', 'feat']].groupby('feat').mean().reset_index()
    mean_gain.index.name = 'feat'
    plt.figure(figsize=(6, 9))
    sns.barplot(x='gain', y='feat', data=mean_gain.sort_values('gain', ascending=False))
    plt.tight_layout()
    plt.savefig('imps.png')

importances = pd.DataFrame()

for i, (_train, _eval) in enumerate(folds.split(train.values, y_tgt)):

    # Setup fold data
    x_all = train[train_cols].values
    x_train, y_train = x_all[_train], y_tgt[_train]
    x_eval, y_eval = x_all[_eval], y_tgt[_eval]

    # Setup multiclass LGBM
    bst = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        objective='multiclass',
        class_weight=class_weights,
        silent=True,
        importance_type='gain',
    )

    # Train bst
    bst.fit(
        X=x_train,
        y=y_train,
        eval_set=[(x_eval, y_eval)],
        eval_names=['\ntrain_set', '\neval_set'],
        eval_class_weight=[class_weights, class_weights],
        eval_metric=lgbm_loss_wrapper,
        early_stopping_rounds=5,
        verbose=True,
    )

    # Print fold train, val loss
    y_pred_train, y_pred_eval = bst.predict_proba(x_train), bst.predict_proba(x_eval)

    print('\nBooster iteration',i,'*'*20)
    print('\nTrain custom loss : ', loss_wrapper(y_true=y_train, y_pred=y_pred_train)[1])
    print('Eval custom loss: ', loss_wrapper(y_true=y_eval, y_pred=y_pred_eval)[1])
    print('\n'+'*'*40+'\n')

    # Importance analysis
    imp_df = pd.DataFrame()
    imp_df['feat'] = train_cols
    imp_df['gain'] = bst.feature_importances_
    imp_df['fold'] = i
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    # Print true vs pred class dist in train and eval - Obsolete
    # for s, preds, truths in zip(['train', 'eval'], [y_pred_train, y_pred_eval], [y_train, y_eval]):
    #     y_pred_1d = pd.Series(class_codes[np.argmax(preds, axis=1)])
    #     y_true_1d = pd.Series(truths)
    #     print('\n'+'*'*20)
    #     print(s)
    #     print('*'*20)
    #     print('\nTrue dist:')
    #     print(y_true_1d.value_counts())
    #     print('\nPred dist:')
    #     print(y_pred_1d.value_counts())

save_importances(importances)