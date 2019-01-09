import numpy as np
import pandas as pd

import time, datetime, os, glob

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from utils.misc_utils import plot_confusion_matrix, save_submission


'''
KNN Class definition
'''

class KNNModel:
    # Constructor
    def __init__(self, train, test, y_tgt, selected_cols, output_dir):

        # Input control
        if train is None:
            raise ValueError('Error initializing KNN - must provide at least a training set')
        is_train_only = True if test is None else False

        # dataset
        self.train = train
        self.test = test
        self.y_tgt = y_tgt
        self.selected_cols = selected_cols

        # other params
        self.output_dir = output_dir
        self.class_weights_dict = {
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

        self.models = []
        self.fold_val_losses = []

        '''
        Input scaling and additional preprocessing
        '''

        subset_train = self.train[self.selected_cols].replace([-np.inf, np.inf], np.nan)
        train_mean = subset_train.mean(axis=0, skipna=True)
        subset_train.fillna(train_mean, inplace=True)

        # We'll do a special scaling for is_galactic flag
        is_gal_col_num = list(subset_train.columns).index('is_galactic')

        self.x_all = subset_train.values
        is_gal_col = np.copy(self.x_all[:,is_gal_col_num])
        is_gal_col *= 1000 # To effectively separate galactic from extra galactic

        ss = StandardScaler()
        ss.fit(self.x_all)
        self.x_all = ss.transform(self.x_all)
        self.x_all[:, is_gal_col_num] = is_gal_col

        if not is_train_only:
            subset_test = self.test[self.selected_cols].replace([-np.inf, np.inf], np.nan)
            subset_test.fillna(train_mean, inplace=True)
            self.x_test = subset_test.values

            is_gal_col = np.copy(self.x_test[:, is_gal_col_num])
            is_gal_col *= 1000

            self.x_test = ss.transform(self.x_test)
            self.x_test[:, is_gal_col_num] = is_gal_col

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
        class_avg_losses = np.sum(y_true[:, valid] * y_pred[:, valid], axis=0) / np.sum(y_true[:, valid], axis=0)
        loss = -np.sum(class_avg_losses * self.weights[valid]) / np.sum(self.weights[valid])

        return loss

    # Main methods
    def fit(self, params):

        '''
        Setup CV
        '''

        # CV cycle collectors
        y_oof = np.zeros(self.y_tgt_oh.shape)

        # Setup stratified CV
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt, self.y_tgt)):
            print(f'>   KNN : Computing fold number {i} . . .')

            # Setup fold data
            x_train, y_train = self.x_all[_train], self.y_tgt[_train]
            x_eval, y_eval = self.x_all[_eval], self.y_tgt_oh[_eval]

            '''
            Model setup and fit
            '''

            knn = KNeighborsClassifier(
                n_neighbors=params['n_neighbors'],
                weights=params['weight'],
                n_jobs=8,
                algorithm='ball_tree',
                #leaf_size=4,
            )
            knn.fit(x_train, y_train)

            '''
            Model prediction on oof
            '''

            y_oof[_eval, :] = knn.predict_proba(x_eval)
            val_loss = self.weighted_average_crossentropy_numpy(y_eval, y_oof[_eval, :])
            self.fold_val_losses.append(val_loss)
            print(f'>    KNN : Fold val loss : {val_loss:.4f}')

            self.models.append(knn)

        print('>    KNN : Remote CV results : ')
        print(pd.Series(self.fold_val_losses).describe())

    def predict(self, iteration_name, predict_test=True, save_preds=True, produce_sub=False, save_confusion=True):

        if not self.models:
            raise ValueError('Must fit models before predicting')

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
            print(f'>   KNN : Predicting on fold number {i} . . .')

            # Setup fold data
            x_eval, y_eval = self.x_all[_eval], self.y_tgt_oh[_eval]

            # Train predictions (oof)
            y_oof[_eval, :] = self.models[i].predict_proba(x_eval)

            # Test predictions
            if predict_test:
                y_test += self.models[i].predict_proba(self.x_test) / num_folds

            val_loss = self.weighted_average_crossentropy_numpy(y_eval, y_oof[_eval, :])
            self.fold_val_losses.append(val_loss)
            print(f'>   KNN : Fold val loss : {val_loss:.4f}')

        final_name = f'knn_{iteration_name}_{np.mean(self.fold_val_losses):.4f}'

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

