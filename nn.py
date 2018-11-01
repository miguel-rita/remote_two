import numpy as np
import pandas as pd

import time, datetime, os

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import utils.preprocess as utils




# Get preprocessed metadata
train_meta, test_meta, y_tgt, train_cols = utils.prep_data()

# Get data and concat to metadata
train_feats = pd.read_hdf('data/training_feats/training_set_feats_first_from_grouped_plus_detected_three.h5', mode='r')
test_feats = pd.read_hdf('data/test_feats/test_set_feats_std.h5', mode='r')

make_sub = False

train_cols.extend(list(train_feats.iloc[:,1:].columns)) # Jump object_id

train = pd.merge(
    train_meta,
    train_feats,
    how='outer',
    on='object_id'
)
test = pd.merge(
    test_meta,
    test_feats,
    how='outer',
    on='object_id'
)





# Standard input scaling

# Take care of NAs (mostly from features)
subset_train = train[train_cols].replace([-np.inf, np.inf], np.nan)
train_mean = subset_train.mean(axis=0)
subset_train.fillna(train_mean, inplace=True)
x_all = subset_train.values

ss = StandardScaler()
ss.fit(x_all)
x_all = ss.transform(x_all)

if make_sub:
    subset_test = test[train_cols].replace([-np.inf, np.inf], np.nan)
    subset_test.fillna(train_mean, inplace=True)
    x_test = subset_test.values
    x_test = ss.transform(x_test)







# Setup everything weight-related, y_tgt OH encoding

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

# Sorted class weights
weights = np.array([class_weights[i] for i in range(len(class_weights))])

# Encode target labels
for i,tgt in enumerate(y_tgt):
    y_tgt[i] = label_encode[tgt]

# Setup OH target
y_tgt_oh = np.eye(len(weights))[y_tgt]

# Setup magic weights
magic_weights = class_weights






'''
Auxiliary funcs definitions
'''

def build_model(dropout_rate=0.25, activation='relu'):
    start_neurons = 512
    
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=x_all.shape[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 2, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 4, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 8, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(len(weights), activation='softmax'))
    return model






# Setup stratified CV
num_folds = 5
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

# Agg vars
nns = []
fold_val_losses = []





for i, (_train, _eval) in enumerate(folds.split(y_tgt, y_tgt)):

    '''
    Model setup
    '''

    # Setup fold data
    x_train, y_train = x_all[_train], y_tgt_oh[_train]
    x_eval, y_eval = x_all[_eval], y_tgt_oh[_eval]

    # Instantiate model
    nn = build_model(dropout_rate=0.25, activation='relu')

    # Compile model
    nn.compile(
        optimizer='SGD',
        loss='categorical_crossentropy',
    )

    '''
    Model fit
    '''

    # Fit overall definitions
    batch_size = 512
    num_epochs = 100000

    hist = nn.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        class_weight=magic_weights,
        validation_data=(x_eval, y_eval),
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=5,
                mode='min',
            )
        ],
        verbose=2,
    )

    nns.append(nn)
    val_loss = hist.history['val_loss'][-1]
    fold_val_losses.append(val_loss)





'''
Model save
'''

# Setup save dir
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
model_name = 'naive_nn'
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
        fmt=['%d'] + ['%.4f'] * num_classes,
        delimiter=',',
        header=h,
        comments='',
    )


# Make sub by averaging preds from all fold models
if make_sub:
    y_preds = []
    for nn in nns:
        y_preds.append(nn.predict(x_test, batch_size=int(x_test.shape[0]/5)))
    y_sub = np.mean(y_preds, axis=0)

    print(f'>   nn : Saving sub . . .')
    save_submission(
        y_sub,
        sub_name=f'./subs/sub_nn_const99_{np.mean(fold_val_losses):.4f}.csv',
        rs_bins=test['rs_bin'].values
    )

print(f'>   nn : Done')