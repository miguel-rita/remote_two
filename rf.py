import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix

from utils.misc_utils import plot_confusion_matrix, save_importances, save_submission

'''
RF Class definition
'''

class RfModel:

    # Constructor
    def __init__(self, train, test, y_tgt, selected_cols, output_dir, params):

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
        self.params = params
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

        '''
        NA preprocess for RF
        '''

        # Take care of NAs (mostly from features)
        subset_train = self.train[self.selected_cols].replace([-np.inf, np.inf], np.nan)
        train_mean = subset_train.mean(axis=0)
        subset_train.fillna(train_mean, inplace=True)
        self.x_all = subset_train.values

        if not is_train_only:
            subset_test = self.test[self.selected_cols].replace([-np.inf, np.inf], np.nan)
            subset_test.fillna(train_mean, inplace=True)
            self.x_test = subset_test.values

    # Loss-related methods
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

    # Main methods
    def fit_predict(self, iteration_name, predict_test=True, save_preds=True, produce_sub=False, save_imps=True,
                    save_confusion=True):

        if produce_sub:
            predict_test = True

        # Compute sample weights
        sample_weights = compute_sample_weight('balanced', self.y_tgt)

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
        x_all = self.train[self.selected_cols].values

        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt, self.y_tgt)):

            print(f'>   rf : Computing fold number {i} . . .')

            # Setup fold data
            x_train, y_train = x_all[_train], self.y_tgt[_train]
            x_eval, y_eval = x_all[_eval], self.y_tgt[_eval]
            sample_weights_train, sample_weights_eval = sample_weights[_train], sample_weights[_eval]

            if predict_test:
                x_test = self.test[self.selected_cols].values

            # Setup RF
            rf = RandomForestClassifier(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                max_features=self.params['max_features'],
                class_weight=self.class_weights_dict,
            )

            # Train RF
            rf.fit(
                X=x_train,
                y=y_train,
                sample_weight=sample_weights_train,
            )

            # Store oof preds, eval loss, define iteration name
            y_oof[_eval, :] = rf.predict_proba(x_eval)
            val_loss = self.weighted_mc_crossentropy(y_eval, y_oof[_eval, :])
            eval_losses.append(val_loss)
            print(f'>   rf : Fold val loss : {val_loss:.4f}')

            # Build test predictions
            if predict_test:
                y_test += rf.predict_proba(x_test) / num_folds

            # Importance analysis
            if save_imps:
                imp_df = pd.DataFrame()
                imp_df['feat'] = self.selected_cols
                imp_df['gain'] = rf.feature_importances_
                imp_df['fold'] = i
                imps = pd.concat([imps, imp_df], axis=0, sort=False)

        final_name = f'rf_{iteration_name}_{np.mean(eval_losses):.4f}'

        if save_imps:
            save_importances(imps, filename_='imps/imps_' + final_name)

        print('>    rf : Remote CV results : ')
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
            plot_confusion_matrix(cm, classes=[str(c) for c in self.class_codes],
                                  filename_='confusion/confusion_' + final_name, normalize=True)