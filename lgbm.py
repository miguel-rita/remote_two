import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix

import utils.preprocess as utils
from utils.misc_utils import plot_confusion_matrix, save_importances

'''
LGBM Class definition
'''

class LgbmModel:
    
    # Constructor
    def __init__(self, train, test, y_tgt, selected_cols, output_dir, fit_params):

        # dataset
        self.train = train
        self.test = test
        self.y_tgt = y_tgt
        self.selected_cols = selected_cols

        # other params
        self.output_dir = output_dir
        self.fit_params = fit_params
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

    # Loss-related methods
    def lgbm_loss_wrapper(self, y_true, y_pred):
        '''
        Custom lgbm eval metric - multiclass weighted logloss
        :param y_true: Encoded 1D targets, needs OHE
        :param y_pred: 1D preds class-first according to lgbm doc
        :return:
        '''

        # Reshape preds from 1D lgb class-first format to standard 2D format
        num_classes = np.max(y_true) + 1
        y_pred = np.reshape(y_pred, (y_true.shape[0], num_classes), order='F')

        return ('lgbm_custom_logloss', self.weighted_mc_crossentropy(y_true, y_pred, weighted=True), False)
    def loss_wrapper(self, y_true, y_pred):
        '''
        Custom eval metric, same as above only inputs are in different format
        :param y_true: Unencoded targets, 1D, needs OHE
        :param y_pred: 2D array of probabilities, summing to 1 per row (across classes)
        :return:
        '''

        # Encode targets
        y_true = np.array([self.label_encode[tgt] for tgt in y_true])

        return ('custom_logloss', self.weighted_mc_crossentropy(y_true, y_pred, weighted=True), False)
    def weighted_mc_crossentropy(self, y_true, y_pred, weighted=True):
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
            return - np.sum(self.weights * avg_loss_per_class) / np.sum(self.weights)

        else:
            return - np.sum(y_true * y_pred) / y_true.shape[0]
        
    # Other methods
    def save_submission(self, y_test, sub_name, rs_bins, nrows=None):

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
    def fit_predict(self, iteration_name='v3.6', predict_test=True, save_preds=True, produce_sub=False, save_imps=True, save_confusion=True):

        if produce_sub:
            predict_test = True

        # Compute sample weights
        sample_weights = compute_sample_weight('balanced', self.y_tgt)
    
        '''
        Setup CV
        '''
    
        # CV cycle collectors
        y_oof = np.zeros((self.y_tgt.size, self.weights.size))
        y_test = np.zeros((self.test.shape[0], self.weights.size))
        eval_losses = []
        imps = pd.DataFrame()
    
        # Setup stratified CV
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
        x_all = self.train[self.selected_cols].values
    
        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt, self.y_tgt)):
    
            print(f'>   lgbm : Computing fold number {i} . . .')
    
            # Setup fold data
            x_train, y_train = x_all[_train], self.y_tgt[_train]
            x_eval, y_eval = x_all[_eval], self.y_tgt[_eval]
            sample_weights_train, sample_weights_eval = sample_weights[_train], sample_weights[_eval]
    
            if predict_test:
                x_test = self.test[self.selected_cols].values
    
            # Setup multiclass LGBM
            bst = lgb.LGBMClassifier(
                boosting_type='gbdt',
                num_leaves=self.fit_params['num_leaves'],
                learning_rate=self.fit_params['learning_rate'],
                n_estimators=self.fit_params['n_estimators'],
                objective='multiclass',
                class_weight=self.class_weights,
                reg_alpha=self.fit_params['reg_alpha'],
                reg_lambda=self.fit_params['reg_lambda'],
                silent=self.fit_params['silent'],
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
                eval_class_weight=[self.class_weights],
                eval_metric=self.lgbm_loss_wrapper,
                early_stopping_rounds=15,
                verbose=False,
            )
    
            # Store oof preds, eval loss, define iteration name
            y_oof[_eval,:] = bst.predict_proba(x_eval)
            val_loss = self.loss_wrapper(y_eval, y_oof[_eval,:])[1]
            eval_losses.append(val_loss)
            print(f'>   lgbm : Fold val loss : {val_loss:.4f}')

            # Build test predictions
            if predict_test:
                y_test += bst.predict_proba(x_test) / num_folds
    
            # Importance analysis
            if save_imps:
                imp_df = pd.DataFrame()
                imp_df['feat'] = self.selected_cols
                imp_df['gain'] = bst.feature_importances_
                imp_df['fold'] = i
                imps = pd.concat([imps, imp_df], axis=0, sort=False)

        final_name = f'lgbm_{iteration_name}_{np.mean(eval_losses):.4f}'
        
        if save_imps:
            save_importances(imps, filename_='imps/imps_'+final_name)

        print('>    lgbm : Remote CV results : ')
        print(pd.Series(eval_losses).describe())

        if save_preds:
            class_names = [final_name + '__' + str(c) for c in self.class_codes]

            oof_preds = pd.concat([self.train[['object_id']], pd.DataFrame(y_oof, columns=class_names)], axis=1)
            oof_preds.to_hdf(self.output_dir + f'{final_name}_oof.h5', key='w')

            if predict_test:
                test_preds = pd.concat([self.test[['object_id']], pd.DataFrame(y_test, columns=class_names)], axis=1)
                test_preds.to_hdf(self.output_dir + f'{final_name}_test.h5', key='w')

        if produce_sub:
            self.save_submission(y_test, f'./subs/{final_name}.csv', rs_bins=np.load('data/rs_bins.npy'))

        if save_confusion:
            y_preds = np.argmax(y_oof, axis=1)
            y_preds = np.array([self.class_codes[i] for i in y_preds])
            cm = confusion_matrix(self.y_tgt, y_preds)
            plot_confusion_matrix(cm, classes=[str(c) for c in self.class_codes], filename_='confusion/confusion_'+final_name, normalize=True)