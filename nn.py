import numpy as np
import pandas as pd

import time, datetime, os, tqdm

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix

import utils.preprocess as utils
from utils.misc_utils import plot_confusion_matrix, save_importances

# Seed
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(1)

'''
Auxiliary funcs definitions
'''

def build_model(dropout_rate, activation='relu'):

    # create model
    model = Sequential()
    model.add(Dense(400, input_dim=x_all.shape[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(40, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(14, activation='softmax'))
    return model

def weighted_average_crossentropy_backend(y_true, y_pred):
    '''
    Custom defined weight multiclass cross entropy
    :param y_true: OHE targets
    :param y_pred: 2D array of probabilities, summing to 1 per row (across classes)
    :return: (float) Calculated loss
    '''

    # In ascending class order : 6,15,...,92,95
    _weights = K.constant([
        1.001044,
        2.001886,
        1.001044,
        1.001044,
        1.001044,
        1.,
        1.001044,
        2.007104,
        1.001044,
        1.001044,
        1.001044,
        1.001044,
        1.001044,
        1.001044,
    ], dtype='float32')

    # Clip preds for log safety
    y_pred = K.log(K.clip(y_pred, 1e-15, 1 - 1e-15))

    # Compute loss
    class_avg_losses = K.sum(y_true * y_pred, axis=0) / K.sum(y_true, axis=0)
    loss = -K.sum(class_avg_losses * _weights) / K.sum(_weights)

    return loss

def weighted_average_crossentropy_numpy(y_true, y_pred):
    '''
    Custom defined weight multiclass cross entropy
    :param y_true: OHE targets
    :param y_pred: 2D array of probabilities, summing to 1 per row (across classes)
    :return: (float) Calculated loss
    '''

    # In ascending class order : 6,15,...,92,95
    _weights = np.array([
        1.001044,
        2.001886,
        1.001044,
        1.001044,
        1.001044,
        1.,
        1.001044,
        2.007104,
        1.001044,
        1.001044,
        1.001044,
        1.001044,
        1.001044,
        1.001044,
    ], dtype='float32')

    # Clip preds for log safety
    y_pred = np.log(np.clip(y_pred, 1e-15, 1 - 1e-15))

    # Compute loss
    class_avg_losses = np.sum(y_true * y_pred, axis=0) / np.sum(y_true, axis=0)
    loss = -np.sum(class_avg_losses * _weights) / np.sum(_weights)

    return loss

def save_submission(y_test, sub_name, rs_bins, nrows=None):

    # Get submission header
    col_names = list(pd.read_csv(filepath_or_buffer='data/sample_submission.csv', nrows=1).columns)
    num_classes = len(col_names) - 1

    # Get test ids
    object_ids = pd.read_csv(filepath_or_buffer='data/test_set_metadata.csv', nrows=nrows, usecols=['object_id']).values.astype(int)
    num_ids = object_ids.size

    # Class 99 adjustment - remember these are conditional probs on redshift
    c99_bin0_prob = 0.021019
    c99_bin1_9_prob = 0.102627

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
    'data/training_feats/training_set_feats_r3_m-feats_weighted_v1.h5',
    'data/training_feats/training_set_feats_r3_t-feats_v1.h5',
    'data/training_feats/training_set_feats_r3_d-feats_v1.h5',
    'data/training_feats/training_set_feats_r3_slope-feats_v1.h5',
    'data/training_feats/training_set_feats_r3_e-feats_v1.h5',
    # 'data/training_feats/training_set_feats_r2_v7.h5',
    # 'data/training_feats/training_set_feats_r2_slope_v2.h5',
    # 'data/training_feats/training_set_feats_r2_err_v1.h5',
    # 'data/training_feats/training_set_feats_r2_exp.h5',
]
test_feats_list = [
    # 'data/test_feats/test_set_feats_std.h5'
    'data/test_feats/test_set_feats_r2_v7.h5',
    'data/test_feats/test_set_feats_r2_slope_v2.h5',
]

train, test, y_tgt, train_cols = utils.prep_data(train_feats_list, test_feats_list)

produce_sub = False
sub_name = 'v1.0'

'''
Class weights
'''

class_weights_dict = {
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

# Encode target labels
for i,tgt in enumerate(y_tgt):
    y_tgt[i] = label_encode[tgt]

# Setup OH target
y_tgt_oh = np.eye(len(weights))[y_tgt]

# Compute sample weights
sample_weights = compute_sample_weight('balanced', y_tgt)

'''
Input scaling and additional preprocess for nn
'''

# Take care of NAs (mostly from features)
subset_train = train[train_cols].replace([-np.inf, np.inf], np.nan)
train_mean = subset_train.mean(axis=0)
subset_train.fillna(train_mean, inplace=True)
x_all = subset_train.values

ss = StandardScaler()
ss.fit(x_all)
x_all = ss.transform(x_all)

if produce_sub:
    subset_test = test[train_cols].replace([-np.inf, np.inf], np.nan)
    subset_test.fillna(train_mean, inplace=True)
    x_test = subset_test.values
    x_test = ss.transform(x_test)

'''
Setup CV
'''

# CV cycle collectors
nns = []
fold_val_losses = []
y_oof = np.zeros(y_tgt_oh.shape)

# Setup stratified CV
num_folds = 5
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

for i, (_train, _eval) in enumerate(folds.split(y_tgt, y_tgt)):

    print(f'>   nn : Computing fold number {i} . . .')

    '''
    Model setup
    '''

    # Setup fold data
    x_train, y_train = x_all[_train], y_tgt_oh[_train]
    x_eval, y_eval = x_all[_eval], y_tgt_oh[_eval]
    sample_weights_fold = sample_weights[_train]

    # Instantiate model
    nn = build_model(dropout_rate=0.25, activation='relu')

    # Compile model
    nn.compile(
        optimizer=optimizers.SGD(lr=0.03, momentum=0, decay=0, nesterov=False),
        loss='categorical_crossentropy',
    )

    '''
    Model fit
    '''

    # Fit overall definitions
    batch_size = 256
    num_epochs = 10000

    hist = nn.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_eval, y_eval),
        sample_weight=sample_weights_fold,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=10,
                mode='min',
            )
        ],
        verbose=0,
    )

    # Debug outputs
    # get_layer_outs = K.function([nn.layers[0].input, K.learning_phase()],
    #                                   [layer.output for layer in nn.layers])
    # layer_output = get_layer_outs([x_eval, 1])
    # _w = nn.layers[0].get_weights()

    y_oof[_eval, :] = nn.predict(x_eval, batch_size=10000)
    val_loss = weighted_average_crossentropy_numpy(y_eval, y_oof[_eval, :])
    fold_val_losses.append(val_loss)
    print(f'>    nn : Fold val loss : {val_loss:.4f}')

    nns.append(nn)

print('>    nn : Remote CV results : ')
print(pd.Series(fold_val_losses).describe())


'''
Model save
'''

# Setup save dir
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
model_name = 'nn_v1'
mean_loss = np.mean(fold_val_losses)
model_name += '__'+timestamp+'__'+f'{mean_loss:.4f}'
os.mkdir(os.getcwd() + '/models/' + model_name)

# Save model
for i,nn in enumerate(nns):
    fold_name = f'fold{i:d}'
    fold_loss = fold_val_losses[i]
    filepath = os.getcwd() + '/models/' +model_name + '/' + fold_name + f'__{fold_loss:.4f}.h5'
    nn.save(filepath=filepath)


'''
Submission creation
'''

# Make sub by averaging preds from all fold models
if produce_sub:
    y_preds = []
    for nn in tqdm.tqdm(nns, total=len(nns)):
        y_preds.append(nn.predict(x_test, batch_size=int(x_test.shape[0]/500)))
    y_sub = np.mean(y_preds, axis=0)

    print(f'>   nn : Saving sub . . .')
    save_submission(
        y_sub,
        sub_name=f'./subs/nn_{sub_name}_{np.mean(fold_val_losses):.4f}.csv',
        rs_bins=test['rs_bin'].values
    )

print(f'>   nn : Done')