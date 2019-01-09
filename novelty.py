import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing, impute
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_recall_curve

'''
Experimental file for unsupervised novelty detection applied to class 99
'''

# Load training feats
train = pd.read_hdf('data/train_feats_od.h5')

# Load tgt and add to feats
train['tgt'] = np.load('data/target_col.npy')

# Lets first try novelty detection on a known class eg. 95

# First, build the training and test sets from one start kfold
y_tgt = train['tgt'].values
num_folds = 8
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
_train, _eval = next(folds.split(y_tgt, y_tgt))
test_X = train.iloc[_eval]

# Build train data - test class eg. 95 (will remain out of train data)
TEST_CLASS = 90
train_X = train.iloc[_train]
pure_mask = train_X['tgt'] != TEST_CLASS
train_X = train_X[pure_mask]

# Get balanced weights
w = compute_sample_weight('balanced', y_tgt)
train_w = w[_train]
train_w = train_w[pure_mask]

# Preprocessing
X_train = train_X.values[:, :-1]
X_test = test_X.values[:, :-1]

# Scale
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Impute
imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

# Fit oneclass SVM
nu = train['tgt'].value_counts()[TEST_CLASS] / train.shape[0]
clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.5)

clf.fit(X_train, sample_weight=train_w)

# Predict on train and test set
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Setup target vectors
y_train = np.copy(train_X.values[:, -1])
y_train[y_train == TEST_CLASS] = -1
y_train[np.logical_and(y_train != TEST_CLASS, y_train != -1)] = 1

y_test = np.copy(test_X.values[:, -1])
y_test[y_test == TEST_CLASS] = -1
y_test[np.logical_and(y_test != TEST_CLASS, y_test != -1)] = 1


# Compute precision / recall for train and test, at different thresholds

def get_stats(y_true, y_pred, pos_label):
    '''
    return tp, fp, fn, tn tuple
    '''

    pmask = y_true == pos_label  # Positive mask on truth
    ppmask = y_pred == pos_label # Pos mask on label
    nmask = np.logical_not(pmask)  # negative mask on truth
    nnmask = np.logical_not(ppmask) # neg mask on pred

    tp = np.sum(y_true[pmask] == y_pred[pmask])
    fp = np.sum(y_true[ppmask] != y_pred[ppmask])
    tn = np.sum(y_true[nmask] == y_pred[nmask])
    fn = np.sum(y_true[nnmask] != y_pred[nnmask])

    return tp, fp, fn, tn


for name, iset, truth in zip(['test'], [X_test], [y_test]):

    # Get scoring function
    y_pred_raw = clf.decision_function(iset)

    # Compute precision / recall for each thresh
    for thresh in np.linspace(np.min(y_pred_raw), np.max(y_pred_raw), 15):
        y_pred = np.copy(y_pred_raw)
        y_pred[y_pred_raw <= thresh] = -1
        y_pred[y_pred_raw > thresh] = 1

        # Get stats
        tp, fp, tn, fn = get_stats(truth, y_pred, pos_label=-1)

        # Compute precision and recall
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)

        # Compute fraction of positives
        p_frac = (tp + fp) / (tp + fp + tn + fn)

        # Print results
        print(f'> {name}, threshold {thresh:.2f} : Precision = {prec:.2f}, Recall = {recall:.2f}, with flagged frac = {p_frac:.2f}')