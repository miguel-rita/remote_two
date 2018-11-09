import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix

import utils.preprocess as utils
from utils.misc_utils import plot_confusion_matrix, save_importances

'''
Misc. utils
'''

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

def save_submission(y_test, sub_name, rs_bins, nrows=None):

    # Get submission header
    col_names = list(pd.read_csv(filepath_or_buffer='data/sample_submission.csv', nrows=1).columns)
    num_classes = len(col_names) - 1

    # Get test ids
    object_ids = pd.read_csv(filepath_or_buffer='data/test_set_metadata.csv', nrows=nrows, usecols=['object_id']).values.astype(int)
    num_ids = object_ids.size

    # Class 99 adjustment - remember these are conditional probs on redshift
    c99_bin0_prob = 0.02
    c99_bin1_9_prob = 0.13

    c99_probs = np.zeros((y_test.shape[0],1))
    c99_probs[rs_bins==0] = c99_bin0_prob
    c99_probs[rs_bins!=0] = c99_bin1_9_prob
    y_test[rs_bins==0] *= (1 - c99_bin0_prob)
    y_test[rs_bins!=0] *= (1 - c99_bin1_9_prob)

    sub = np.hstack([object_ids, y_test, c99_probs])

    h = ''
    for s in col_names:
        h += s + ','
    h = h[:-1]

    # Write to file
    np.savetxt(
        fname=sub_name,
        X=sub,
        fmt=['%d'] + ['%.6f'] * num_classes,
        delimiter=',',
        header=h,
        comments='',
    )

'''
Load and preprocess data
'''

train_feats_list = [
    'data/training_feats/training_set_feats_r3_slope-feats_v2.h5',
    'data/training_feats/training_set_feats_r3_m-feats_v3.h5',
    'data/training_feats/training_set_feats_r3_t-feats_v1.h5',
    'data/training_feats/training_set_feats_r3_d-feats_v1.h5',
    'data/training_feats/training_set_feats_r3_e-feats_v1.h5',
    'data/training_feats/training_set_feats_r3_curve-feats_v1.h5',
    'data/training_feats/training_set_feats_r3_linreg-feats_v1.h5',
]
test_feats_list = [
    # 'data/test_feats/test_set_feats_std.h5'
    'data/test_feats/test_set_feats_r3_slope-feats_v2.h5',
    'data/test_feats/test_set_feats_r3_m-feats_v3.h5',
    'data/test_feats/test_set_feats_r3_t-feats_v1.h5',
    'data/test_feats/test_set_feats_r3_d-feats_v1.h5',
    'data/test_feats/test_set_feats_r3_e-feats_v1.h5',
    'data/test_feats/test_set_feats_r3_curve-feats_v1.h5',
    'data/test_feats/test_set_feats_r3_linreg-feats_v1.h5',

]

train, test, y_tgt, train_cols = utils.prep_data(train_feats_list, test_feats_list)

produce_sub = False
sub_name = 'v3.4'

'''
Class weights
'''

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

# Compute sample weights
sample_weights = compute_sample_weight('balanced', y_tgt)

'''
Setup CV
'''

# CV cycle collectors
importances = pd.DataFrame()
y_preds_oof = np.zeros((y_tgt.size, weights.size))
y_test = np.zeros((test.shape[0], weights.size))
eval_losses = []
bsts = []

# Setup stratified CV
num_folds = 5
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

for i, (_train, _eval) in enumerate(folds.split(y_tgt, y_tgt)):

    print(f'>   lgbm : Computing fold number {i} . . .')

    # Setup fold data
    x_all = train[train_cols].values
    x_train, y_train = x_all[_train], y_tgt[_train]
    x_eval, y_eval = x_all[_eval], y_tgt[_eval]
    sample_weights_train, sample_weights_eval = sample_weights[_train], sample_weights[_eval]

    if produce_sub:
        x_test = test[train_cols].values

    # Setup multiclass LGBM
    bst = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=7,
        learning_rate=0.10,
        n_estimators=10000,
        objective='multiclass',
        class_weight=class_weights,
        reg_alpha=1,
        reg_lambda=5,
        silent=True,
        verbose=-1,
    )

    # Train bst
    bst.fit(
        X=x_train,
        y=y_train,
        sample_weight=sample_weights_train,
        eval_sample_weight=[sample_weights_eval],
        eval_set=[(x_eval, y_eval)],
        eval_names=['\neval_set'],
        eval_class_weight=[class_weights],
        eval_metric=lgbm_loss_wrapper,
        early_stopping_rounds=15,
        verbose=False,
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

# Plot confusion matrix
y_preds = np.argmax(y_preds_oof, axis=1)
y_preds = np.array([class_codes[i] for i in y_preds])
cm = confusion_matrix(y_tgt, y_preds)
plot_confusion_matrix(cm, classes=[str(c) for c in class_codes], normalize=True)

train = pd.concat([train, pd.DataFrame(y_preds_oof, columns=[str(c) for c in class_codes])], axis=1)
train['pred'] = y_preds
train['target'] = y_tgt
train.to_hdf('train_oof.h5', key='w')

if produce_sub:
    save_submission(y_test, f'./subs/lgbm_{sub_name}_{np.mean(eval_losses):.4f}.csv', rs_bins=test['rs_bin'].values)
