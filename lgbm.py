import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix
from utils.misc_utils import plot_confusion_matrix, save_importances, save_submission, mask_samples

'''
LGBM Class definition
'''

class LgbmModel:
    
    # Constructor
    def __init__(self, train, test, y_tgt, selected_cols, output_dir, fit_params, mask_dict=None):

        # dataset
        self.train = train
        self.test = test
        self.y_tgt = y_tgt
        self.selected_cols = selected_cols

        # other params
        self.mask_dict = mask_dict
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


    def weighted_mc_crossentropy(self, y_true, y_pred, weighted=True):
        '''
        Custom defined weight multiclass cross entropy
        :param y_true: Encoded 1D targets, needs OHE
        :param y_pred: 2D array of probabilities, summing to 1 per row (across classes)
        :param weighted: bool - If True all weights will be set to 1
        :return: (float) Calculated loss
        '''

        # Encode targets
        y_true = np.array([self.label_encode[tgt] for tgt in y_true])

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

    def fit_predict(self, iteration_name, predict_test=True, save_preds=True, produce_sub=False, save_imps=True, save_confusion=True):

        if produce_sub:
            predict_test = True

        # Compute sample weights
        if self.mask_dict is not None:
            msk = mask_samples(self.y_tgt, self.mask_dict)
            sample_weights = np.zeros(self.y_tgt.size)
            # sample_weights[msk] = compute_sample_weight('balanced', self.y_tgt[msk])
        else:
            # sample_weights = compute_sample_weight('balanced', self.y_tgt)
            sample_weights = np.zeros(self.y_tgt.size)
            orig_weights = np.ones(shape=self.y_tgt.shape)

            # Bias algorithm towards higher redshifts
            rss = self.train['hostgal_specz'].values

            # Load rs KDE ratios
            # tx, cov_ratio = np.load('data/covariate_tx.npy'), np.load('data/covariate_ratio.npy')
            # orig_weights *= np.interp(rss, tx, cov_ratio)
            # np.save('data/rs_plus2_covar_beta.npy', orig_weights)

            # Load covariate ratios
            # orig_weights *= np.load('data/exp_covar_beta_v1.npy')

            orig_weights *= (1 + 2*rss)

            # Build dict with sum per target
            sum_dict = {}
            for tgt in np.unique(self.y_tgt):
                sum_dict[tgt] = np.sum(orig_weights[self.y_tgt == tgt])

            # Assign balanced weights
            for i, tgt in enumerate(self.y_tgt):
                sample_weights[i] = orig_weights[i] / sum_dict[tgt]

            # Normalize weights
            sample_weights /= np.mean(sample_weights)


        '''
        Setup CV
        '''
    
        # CV cycle collectors
        y_oof = np.zeros((self.y_tgt.size, self.weights.size))
        if predict_test:
            y_test = np.zeros((self.test.shape[0], self.weights.size))
        eval_losses = []
        imps = pd.DataFrame()

        # Setup stratified CV
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
        # self.train[self.selected_cols].to_hdf('data/train_feats_od.h5', key='w')
        # self.test[self.selected_cols].to_hdf('data/test_feats_od.h5', key='w')
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
                min_child_samples=self.fit_params['min_child_samples'],
                silent=self.fit_params['silent'],
                bagging_fraction=self.fit_params['bagging_fraction'],
                bagging_freq=self.fit_params['bagging_freq'],
                bagging_seed=self.fit_params['bagging_seed'],
                verbose=self.fit_params['verbose'],
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
                early_stopping_rounds=15,
                verbose=self.fit_params['verbose'],
            )
    
            # Store oof preds, eval loss, define iteration name
            y_oof[_eval,:] = bst.predict_proba(x_eval)
            val_loss = self.weighted_mc_crossentropy(y_eval, y_oof[_eval,:])
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
            save_submission(y_test, sub_name=f'./subs/{final_name}.csv', rs_bins=np.load('data/rs_bins.npy'))

        if save_confusion:
            y_preds = np.argmax(y_oof, axis=1)
            y_preds = np.array([self.class_codes[i] for i in y_preds])
            cm = confusion_matrix(self.y_tgt, y_preds)
            plot_confusion_matrix(cm, classes=[str(c) for c in self.class_codes], filename_='confusion/confusion_'+final_name, normalize=True)