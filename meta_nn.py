import os
import time, datetime
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import utils.preprocess as utils

# Load data
train, test, y_tgt, train_cols = utils.prep_data()

# Define neural net
eval_losses = []
bsts = []

# Setup stratified CV
num_folds = 5
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

for i, (_train, _eval) in enumerate(folds.split(y_tgt, y_tgt)):

    # Setup fold data
    x_all = train[train_cols].values
    x_train, y_train = x_all[_train], y_tgt[_train]
    x_eval, y_eval = x_all[_eval], y_tgt[_eval]
    x_test = test[train_cols].values

    # Instantiate model
    nn = Sequential(
        [
            Dense(30),
            BatchNormalization(),
            Activation('relu'),
            Dense(1),
            BatchNormalization(),
            Activation('sigmoid'),
        ]
    )

    # Compile model (using bin crossentropy also as testing metric)
    fc_model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',
    )

    # Fit overall definitions
    batch_size = 1024
    num_epochs = 100

    # Fit logistic regression
    hist = fc_model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=2,
                mode='min',
            )
        ],
        verbose=2,
    )

    '''
    3) Model save
    '''

    # Validation loss after training
    val_loss = '{:.4}'.format(hist.history['val_loss'][-1])

    # Grab timestamp string
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    # Save model
    filepath = os.getcwd() + '/saved_models/' + model_name + '__' + timestamp + '__' + val_loss + '.h5'

    fc_model.save(filepath=filepath)