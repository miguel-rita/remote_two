import pandas as pd
import numpy as np
import os, time, tqdm
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import utils.preprocess as utils

# Get metadata
train, test, y_tgt, train_cols = utils.prep_data()

# Get data
produce_sub = True
train_feats = pd.read_hdf('data/train_feats/train_set_feats_v4.h5', mode='r')
test_feats = pd.read_hdf('data/test_feats/test_set_feats_v4.h5', mode='r')
train_cols.extend(list(train_feats.columns))

# Merge
train = pd.merge(
    train,
    train_feats,
    how='outer',
    on='object_id'
)
test = pd.merge(
    test,
    test_feats,
    how='outer',
    on='object_id'
)

# Get sorted class weights
class_weights_dict = {
    #99 : 2.002408,
    95 : 1.001044,
    92 : 1.001044,
    90 : 1.001044,
    88 : 1.001044,
    67 : 1.001044,
    65 : 1.001044,
    64 : 2.007104,
    62 : 1.001044,
    53 : 1.000000,
    52 : 1.001044,
    42 : 1.001044,
    16 : 1.001044,
    15 : 2.001886,
    6 : 1.001044,
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
    eps = np.finfo(float).eps
    y_pred = np.log(np.clip(y_pred, eps, 1 - eps))

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

def save_submission(y_test, sub_name, nrows=None):

    # Get submission header
    col_names = list(pd.read_csv(filepath_or_buffer='data/sample_submission.csv', nrows=1).columns)
    num_classes = len(col_names) - 1

    # Get test ids
    object_ids = pd.read_csv(filepath_or_buffer='data/test_set_metadata.csv', nrows=nrows, usecols=['object_id']).values.astype(int)
    num_ids = object_ids.size

    # Naive sub
    obj_99_prob = np.ones((num_ids, 1)) * 1/10
    factor = 1-1/10
    sub = np.hstack([object_ids, y_test*factor, obj_99_prob])

    h = ''
    for s in col_names:
        h += s + ','
    h = h[:-1]

    # Write to file
    np.savetxt(
        fname=sub_name,
        X=sub,
        fmt=['%d'] + ['%.3f'] * num_classes,
        delimiter=',',
        header=h,
        comments='',
    )

# CV cycle collectors
importances = pd.DataFrame()
y_preds_oof = np.zeros((y_tgt.size, weights.size))
y_test = np.zeros((test.shape[0], weights.size))
eval_losses = []
bsts = []

# Setup stratified CV
num_folds = 8
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)



for i, (_train, _eval) in enumerate(folds.split(y_tgt, y_tgt)):

    print('On fold ',i)

    # Setup fold data
    x_all = train[train_cols].values
    x_train, y_train = x_all[_train], y_tgt[_train]
    x_eval, y_eval = x_all[_eval], y_tgt[_eval]

    if produce_sub:
        x_test = test[train_cols].values

    # Setup multiclass LGBM
    bst = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=4,
        learning_rate=0.03,
        n_estimators=10000,
        objective='multiclass',
        class_weight=class_weights,
        reg_alpha=0.01,
        reg_lambda=0.01,
        silent=False,
    )

    # Train bst
    bst.fit(
        X=x_train,
        y=y_train,
        eval_set=[(x_eval, y_eval)],
        eval_names=['\neval_set'],
        eval_class_weight=[class_weights],
        eval_metric=lgbm_loss_wrapper,
        early_stopping_rounds=15,
        verbose=True,
    )

    # Store oof preds, eval loss
    y_preds_oof[_eval,:] = bst.predict_proba(x_eval)
    eval_losses.append(loss_wrapper(y_eval, y_preds_oof[_eval,:])[1])
    bsts.append(bst)

    # Build test predictions
    if produce_sub:
        y_test += bst.predict_proba(x_test) / num_folds

    # Importance analysis
    imp_df = pd.DataFrame()
    imp_df['feat'] = train_cols
    imp_df['gain'] = bst.feature_importances_
    imp_df['fold'] = i
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

print('Remote CV results : ')
print(pd.Series(eval_losses).describe())
save_importances(importances)

if produce_sub:
    save_submission(y_test, f'./subs/sub_{np.mean(eval_losses):.4f}.csv')
