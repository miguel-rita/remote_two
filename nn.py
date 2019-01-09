import numpy as np
import pandas as pd

import time, datetime, os, glob

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers, backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix

from utils.misc_utils import plot_confusion_matrix, save_submission

# Seed
from numpy.random import seed
from tensorflow import set_random_seed

'''
MLP Class definition
'''

class MlpModel:

    # Constructor
    def __init__(self, train, test, y_tgt, selected_cols, output_dir):

        # Input control
        if train is None:
            raise ValueError('Error initializing MLP - must provide at least a training set')
        is_train_only = True if test is None else False

        # dataset
        self.train = train
        self.test = test
        self.y_tgt = y_tgt
        self.selected_cols = selected_cols

        # other params
        self.output_dir = output_dir
        self.class_weights_dict = {
            # 99 : 2.002408,
            95: 1.001044,
            92: 1.001044,
            90: 1.001044,
            88: 1.001044,
            67: 1.001044,
            65: 1.001044,
            64: 2.007104,
            62: 1.001044,
            53: 1.000000,
            52: 1.001044,
            42: 1.001044,
            16: 1.001044,
            15: 2.001886,
            6: 1.001044,
        }
        self.class_codes = np.unique(list(self.class_weights_dict.keys()))
        self.class_weights = {i: self.class_weights_dict[c] for i, c in enumerate(self.class_codes)}
        self.label_encode = {c: i for i, c in enumerate(self.class_codes)}
        self.weights = np.array([self.class_weights[i] for i in range(len(self.class_weights))])

        # Encode target labels
        for i, tgt in enumerate(self.y_tgt):
            self.y_tgt[i] = self.label_encode[tgt]

        # Setup OH target
        self.y_tgt_oh = np.eye(len(self.weights))[self.y_tgt]

        # Compute sample weights
        # self.sample_weights = compute_sample_weight('balanced', self.y_tgt)

        self.sample_weights = np.zeros(self.y_tgt.size)
        orig_weights = np.ones(shape=self.y_tgt.shape)

        # Bias algorithm towards higher redshifts
        rss = self.train['hostgal_specz'].values
        orig_weights *= (1 + 2*rss)

        # Build dict with sum per target
        sum_dict = {}
        for tgt in np.unique(self.y_tgt):
            sum_dict[tgt] = np.sum(orig_weights[self.y_tgt == tgt])

        # Assign balanced weights
        for i, tgt in enumerate(self.y_tgt):
            self.sample_weights[i] = orig_weights[i] / sum_dict[tgt]

        # Normalize weights
        self.sample_weights /= np.mean(self.sample_weights)

        self.models = []
        self.fold_val_losses = []

        '''
        Input scaling and additional preprocess for nn
        '''

        # nan_cols = [
        #     'linreg_b1_0_110_band2',
        #     'linreg_b1_0_110_band3',
        #     'linreg_b1_0_110_band4',
        #     'abs_magnitude_max_0',
        #     'abs_magnitude_max_1',
        #     'abs_magnitude_max_2',
        #     'abs_magnitude_max_3',
        #     'abs_magnitude_max_4',
        #     'abs_magnitude_max_5',
        #     'absmagmax_ratio_bands_2_3',
        #     'absmagmax_ratio_bands_2_4',
        #     'absmagmax_ratio_bands_2_5',
        #     'absmagmax_ratio_bands_3_4',
        #     'absmagmax_ratio_bands_3_5',
        #     'absmagmax_ratio_bands_4_5',
        #     'spike_back_mean',
        #     'spike_front_mean',
        # ]
        # for nc in nan_cols:
        #     selected_cols.remove(nc)


        # trash_subset_train = self.train[self.selected_cols].to_hdf('data/inputnnexp.h5',key='w')
        # Take care of NAs (mostly from features)

        subset_train = self.train[self.selected_cols].replace([-np.inf, np.inf], np.nan)
        train_mean = subset_train.mean(axis=0, skipna=True)
        subset_train.fillna(train_mean, inplace=True)
        self.x_all = subset_train.values

        ss = StandardScaler()
        ss.fit(self.x_all)
        self.x_all = ss.transform(self.x_all)

        if not is_train_only:
            subset_test = self.test[self.selected_cols].replace([-np.inf, np.inf], np.nan)
            subset_test.fillna(train_mean, inplace=True)
            self.x_test = subset_test.values
            self.x_test = ss.transform(self.x_test)

    # Loss-related methods
    def weighted_average_crossentropy_backend(self, y_true, y_pred):
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
        loss = -K.sum(class_avg_losses * self.weights) / K.sum(self.weights)

        return loss
    def weighted_average_crossentropy_numpy(self, y_true, y_pred):
        '''
        Custom defined weight multiclass cross entropy
        :param y_true: OHE targets
        :param y_pred: 2D array of probabilities, summing to 1 per row (across classes)
        :return: (float) Calculated loss
        '''

        # Clip preds for log safety
        y_pred = np.log(np.clip(y_pred, 1e-15, 1 - 1e-15))

        # Mask incase a class is absent from train set
        valid = np.sum(y_true, axis=0) > 0

        # Compute loss
        class_avg_losses = np.sum(y_true[:,valid] * y_pred[:,valid], axis=0) / np.sum(y_true[:,valid], axis=0)
        loss = -np.sum(class_avg_losses * self.weights[valid]) / np.sum(self.weights[valid])

        return loss

    # State-related methods
    def save(self, model_name):
        '''
        Model save
        '''

        # Setup save dir
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        mean_loss = np.mean(self.fold_val_losses)
        model_name += '__' + timestamp + '__' + f'{mean_loss:.4f}'
        os.mkdir(os.getcwd() + '/models/' + model_name)

        # Save model
        for i, nn in enumerate(self.models):
            fold_name = f'fold{i:d}'
            fold_loss = self.fold_val_losses[i]
            filepath = os.getcwd() + '/models/' + model_name + '/' + fold_name + f'__{fold_loss:.4f}.h5'
            nn.save(filepath=filepath)
    def load(self, models_rel_dir):
        '''
        Load pretrained nets into memory
        '''
        nn_names = glob.glob(os.getcwd() + '/' + models_rel_dir + '/*.h5')
        nn_names.sort()
        self.models.extend([load_model(os.getcwd() + '/' + models_rel_dir + f'/fold{i}__' + n.split('__')[-1]) for i,n in enumerate(nn_names)])

    def build_model(self, layer_dims, dropout_rate, activation='relu'):
        # create model
        model = Sequential()

        if len(layer_dims)<1:
            raise ValueError('Mlp must have at least one layer')

        # first layer, smaller dropout
        model.add(Dense(layer_dims[0], input_dim=self.x_all.shape[1], activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate / 4))

        # further layers
        for ld in layer_dims[1:]:
            model.add(Dense(ld, activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(Dense(14, activation='softmax'))
        return model

    # Main methods
    def fit(self, params):
        
        '''
        Setup CV
        '''
        
        # CV cycle collectors
        nns = []
        fold_val_losses = []
        y_oof = np.zeros(self.y_tgt_oh.shape)
        
        # Setup stratified CV
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
        
        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt, self.y_tgt)):
        
            print(f'>   nn : Computing fold number {i} . . .')
        
            # Setup fold data
            x_train, y_train = self.x_all[_train], self.y_tgt_oh[_train]
            x_eval, y_eval = self.x_all[_eval], self.y_tgt_oh[_eval]
            sample_weights_fold = self.sample_weights[_train]

            '''
            Model setup
            '''

            # Instantiate model
            nn = self.build_model(layer_dims=params['layer_dims'], dropout_rate=params['dropout_rate'], activation='relu')
        
            # Compile model
            nn.compile(
                optimizer=optimizers.SGD(lr=params['lr'], momentum=0, decay=0, nesterov=False),
                loss='categorical_crossentropy',
            )
        
            '''
            Model fit
            '''
        
            # Fit overall definitions
            batch_size = params['batch_size']
            num_epochs = params['num_epochs']
        
            hist = nn.fit(
                x=x_train,
                y=y_train,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(x_eval, y_eval),
                sample_weight=sample_weights_fold,
                class_weight=self.weights,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        mode='min',
                    )
                ],
                verbose=params['verbose'],
            )

            self.models.append(nn)

            # Debug outputs
            # get_layer_outs = K.function([nn.layers[0].input, K.learning_phase()],
            #                                   [layer.output for layer in nn.layers])
            # layer_output = get_layer_outs([x_eval, 1])
            # _w = nn.layers[0].get_weights()

            y_oof[_eval, :] = nn.predict(x_eval, batch_size=10000)
            val_loss = self.weighted_average_crossentropy_numpy(y_eval, y_oof[_eval, :])
            self.fold_val_losses.append(val_loss)
            print(f'>    nn : Fold val loss : {val_loss:.4f}')

        print('>    nn : Remote CV results : ')
        print(pd.Series(self.fold_val_losses).describe())
    def predict(self, iteration_name, predict_test=True, save_preds=True, produce_sub=False, save_confusion=True):

        if not self.models:
            raise ValueError('Must fit or load models before predicting')

        if produce_sub:
            predict_test = True

        '''
        Setup CV
        '''

        y_oof = np.zeros(self.y_tgt_oh.shape)
        if predict_test:
            y_test = np.zeros((self.test.shape[0], self.weights.size))

        # Setup stratified CV
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt, self.y_tgt)):
            print(f'>   nn : Predicting on fold number {i} . . .')

            # Setup fold data
            x_eval, y_eval = self.x_all[_eval], self.y_tgt_oh[_eval]

            # Train predictions (oof)
            y_oof[_eval, :] = self.models[i].predict(x_eval, batch_size=10000)

            # Test predictions
            if predict_test:
                y_test += self.models[i].predict(self.x_test, batch_size=10000) / num_folds

            val_loss = self.weighted_average_crossentropy_numpy(y_eval, y_oof[_eval, :])
            self.fold_val_losses.append(val_loss)
            print(f'>   nn : Fold val loss : {val_loss:.4f}')

        final_name = f'mlp_{iteration_name}_{np.mean(self.fold_val_losses):.4f}'

        if produce_sub:
            save_submission(y_test, sub_name=f'./subs/{final_name}.csv', rs_bins=self.test['rs_bin'].values)

        if save_confusion:
            y_preds = np.argmax(y_oof, axis=1)
            cm = confusion_matrix(self.y_tgt, y_preds, labels=np.unique(self.y_tgt))
            plot_confusion_matrix(cm, classes=[str(c) for c in self.class_codes[np.unique(self.y_tgt)]],
                                  filename_='confusion/confusion_' + final_name, normalize=True)

        if save_preds:

            class_names = [final_name + '__' + str(c) for c in self.class_codes]

            oof_preds = pd.concat([self.train[['object_id']], pd.DataFrame(y_oof, columns=class_names)], axis=1)
            oof_preds.to_hdf(self.output_dir + f'{final_name}_oof.h5', key='w')

            if predict_test:
                test_preds = pd.concat([self.test[['object_id']], pd.DataFrame(y_test, columns=class_names)], axis=1)
                test_preds.to_hdf(self.output_dir + f'{final_name}_test.h5', key='w')

                return oof_preds, test_preds

