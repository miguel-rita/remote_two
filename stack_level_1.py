import utils.preprocess as utils
import numpy as np
import pandas as pd
import pickle
from lgbm import LgbmModel
from nn import MlpModel
from knn import KNNModel
from svm import SVCModel

def main():

    '''
    Load and preprocess data
    '''

    # Select relevant cached features

    train_feats_list = [
        'data/training_feats/training_set_feats_r7_m-feats_v3.h5',
        # 'data/training_feats/training_set_feats_r7_allm-feats_v1.h5',
        'data/training_feats/training_set_feats_r6_t-feats_v1.h5',
        'data/training_feats/training_set_feats_r6_d-feats_v2.h5',
        'data/training_feats/training_set_feats_r7_linreg-feats_v5.h5',
        'data/training_feats/training_set_feats_r7_expreg-feats_v3.h5',
        # 'data/training_feats/training_set_feats_r6_perc-ratios-feats_v1.h5',
        # 'data/training_feats/training_set_feats_r6_adim-linreg-feats_v1.h5',
        'data/training_feats/training_set_feats_r7_absmag-feats_v4.h5',
        'data/training_feats/training_set_feats_r4_spike-feats_v1.h5',
        'data/training_feats/training_set_feats_r5_linreg-feats_v6_abs.h5'
        # 'data/training_feats/training_set_feats_r6_maxima-feats_v1.h5',
        # 'data/training_feats/training_set_feats_r6_sn-feats_v1.h5',
        # 'data/training_feats/training_set_feats_r6_flag-feats_v1.h5',
        # 'data/training_feats/training_set_feats_r6_shoulder-feats_v1.h5',
        # 'data/training_feats/experimental_sn_v2_tfall.h5',
        # 'data/aug_training_feats/aug_training_set_feats_r4_m-feats_v1.h5',
        # 'data/aug_training_feats/aug_training_set_feats_r4_t-feats_v3.h5',
        # 'data/aug_training_feats/aug_training_set_feats_r4_d-feats_v1.h5',
        # 'data/aug_training_feats/aug_training_set_feats_r4_linreg-feats_v1.h5',
        # 'data/training_feats/training_set_feats_r4_peak-feats-mean15_v4.h5',
        # 'data/training_feats/training_set_feats_r3_slope-feats_v2.h5'
    ]
    test_feats_list = [
        # 'data/test_feats/test_set_feats_std.h5',
        'data/test_feats/test_set_feats_r4_m-feats_v1.h5',
        # 'data/test_feats/test_set_feats_r7_allm-feats_v1.h5',
        'data/test_feats/test_set_feats_r6_t-feats_v1.h5',
        'data/test_feats/test_set_feats_r6_d-feats_v2.h5',
        'data/test_feats/test_set_feats_r7_linreg-feats_v5.h5',
        'data/test_feats/test_set_feats_r7_expreg-feats_v3.h5',
        # 'data/test_feats/test_set_feats_r6_adim-linreg-feats_v1.h5',
        'data/test_feats/test_set_feats_r7_absmag-feats_v4.h5',
        'data/test_feats/test_set_feats_r4_spike-feats_v1.h5',
        'data/test_feats/test_set_feats_r5_linreg-feats_v6_abs.h5',
        # 'data/test_feats/test_set_feats_r6_nnflag-feats_v1.h5',
    ]
    train, test, y_tgt, selected_cols = utils.prep_data(train_feats_list, test_feats_list, augmented=False)

    # train.to_hdf('data/train_small_subset.h5', key='w')
    # np.save('data/target_dummy, y_tgt)

    # Suppress selected cols
    # for b in np.arange(1,5):
    #     selected_cols.remove(f'flux_mean_{b:d}')
    # for b in range(6):
    #     selected_cols.remove(f'flux_max_{b:d}')
    # for b in [0,3,4]:
    #     selected_cols.remove(f'flux_min_{b:d}')
    # for b in [0,1,4]:
    #     selected_cols.remove(f'cross_detected_contrib_{b:d}')
    # for b in [3,4]:
    #     selected_cols.remove(f'flux_std_{b:d}')
    # for b in [4]:
    #     selected_cols.remove(f'cross_band_flux_max_contrib_{b:d}')

    # selected_cols.remove('flux_kurt_0')
    # selected_cols.remove('flux_kurt_1')
    # selected_cols.remove('flux_skew_0')
    # selected_cols.remove('flux_skew_5')

    # selected_cols.remove('absmagmax_ratio_bands_2_4')
    # selected_cols.remove('absmagmax_diff_bands_3_5')

    # Select models to train

    controls = {
        'lgbm-models'   : bool(0),
        'mlp-models'    : bool(1),
        'knn-models'    : bool(0),
        'svm-models'    : bool(0),
    }
    model_name = 'vFFF'

    '''
    LGBM Models
    '''
    if controls['lgbm-models']:

        # No need of this binary flag for tree-based models
        selected_cols.remove('is_galactic')
        # Classifier performance improved by also removing these
        selected_cols.remove('hostgal_photoz')

        lgbm_params = {
            'num_leaves' : 7,
            'learning_rate': 0.10,
            'min_child_samples' : 70,
            'n_estimators': 100,
            'reg_alpha': 1,
            'reg_lambda': 5,
            'bagging_fraction' : 0.70,
            'bagging_freq' : 1,
            'bagging_seed' : 1,
            'silent': -1,
            'verbose': -1,
        }

        lgbm_model_0 = LgbmModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir='./level_1_preds/',
            fit_params=lgbm_params,
            # mask_dict={
            #     90 : (2,0),
            # }
        )

        lgbm_model_0.fit_predict(
            iteration_name=model_name,
            predict_test=False,
            save_preds=True,
            produce_sub=True,
            save_imps=True,
            save_confusion=True
        )

    '''
    MLP models
    '''
    if controls['mlp-models']:

        # Classifier performance improved by also removing these
        selected_cols.remove('hostgal_photoz')

        mlp_output_dir = './level_1_preds/'

        # With Galactic/Extra galactic split

        split_models = False
        if split_models:
            # Split dataset in galactic/extra galactic
            train_gal = train.loc[train['is_galactic'] == 1, :]
            test_gal = test.loc[test['is_galactic'] == 1, :]
            y_tgt_gal = y_tgt[train['is_galactic'] == 1]
            train_gal.drop(columns=['is_galactic'])
            test_gal.drop(columns=['is_galactic'])

            train_extra = train.loc[train['is_galactic'] == 0, :]
            test_extra = test.loc[test['is_galactic'] == 0, :]
            y_tgt_extra = y_tgt[train['is_galactic'] == 0]
            train_extra.drop(columns=['is_galactic'])
            test_extra.drop(columns=['is_galactic'])

            train_gal = train_gal.reset_index(drop=True)
            test_gal = test_gal.reset_index(drop=True)
            train_extra = train_extra.reset_index(drop=True)
            test_extra = test_extra.reset_index(drop=True)

            # There are certain extragalactic-only feats - these must be removed for galactic MLP
            extragalactic_feat_lists = [
                'data/training_feats/training_set_feats_r7_linreg-feats_v5.pkl',
                'data/training_feats/training_set_feats_r7_absmag-feats_v4.pkl',
            ]

            # Assert this list is coeherent with top lists - safety check
            for fl in extragalactic_feat_lists:
                assert fl.split('.pkl')[0] in [ol.split('.h5')[0] for ol in train_feats_list]

            extragal_only_feats = []
            for fl in extragalactic_feat_lists:
                with open(fl, 'rb') as f:
                    extragal_only_feats.extend(pickle.load(f))

            # Remove extragal feats for gal MLP
            selected_cols_gal = selected_cols.copy()
            for ft in extragal_only_feats:
                selected_cols_gal.remove(ft)

            # GALACTIC MLP

            gal_mlp_params = {
                'lr': 0.025,
                'dropout_rate': 0.15,
                'batch_size': 512,
                'num_epochs': 10000,
                'layer_dims': [50,20],
                'verbose': 0,
            }

            gal_mlp_model_0 = MlpModel(
                train=train_gal,
                test=test_gal,
                y_tgt=y_tgt_gal,
                selected_cols=selected_cols_gal,
                output_dir=mlp_output_dir,
            )

            gal_model_name = model_name + '_galactic'

            # gal_mlp_model_0.fit(params=gal_mlp_params)
            # gal_mlp_model_0.save(gal_model_name)
            gal_mlp_model_0.load('models/v8.28_galactic__2018-12-13_23:12:41__0.3063')

            # EXTRAGALACTIC MLP

            extra_mlp_params = {
                'lr': 0.025,
                'dropout_rate': 0.3,
                'batch_size': 512,
                'num_epochs': 10000,
                'layer_dims': [50,20],
                'verbose': 0,
            }

            extra_mlp_model_0 = MlpModel(
                train=train_extra,
                test=test_extra,
                y_tgt=y_tgt_extra,
                selected_cols=selected_cols,
                output_dir=mlp_output_dir,
            )

            extra_model_name = model_name + '_extra'

            # extra_mlp_model_0.fit(params=extra_mlp_params)
            # extra_mlp_model_0.save(extra_model_name)
            extra_mlp_model_0.load('models/v8.28_extra__2018-12-13_22:54:58__1.0686')

            '''
            Predict using mlp models - Combo merge prediction
            '''

            gal_oof, gal_test = gal_mlp_model_0.predict(
                iteration_name=gal_model_name,
                predict_test=True,
                save_preds=True,
                produce_sub=False,
                save_confusion=True
            )

            extra_oof, extra_test = extra_mlp_model_0.predict(
                iteration_name=extra_model_name,
                predict_test=True,
                save_preds=True,
                produce_sub=False,
                save_confusion=True
            )

            # Merge galactic and extra galactic predictions, first adjusting column names
            final_col_names = [cn.replace('extra', 'duo') for cn in list(extra_oof.columns)]
            gal_oof.columns = final_col_names
            extra_oof.columns = final_col_names
            gal_test.columns = final_col_names
            extra_test.columns = final_col_names

            full_oof = pd.concat([gal_oof, extra_oof], axis=0, sort=False).sort_values('object_id')
            full_test = pd.concat([gal_test, extra_test], axis=0, sort=False).sort_values('object_id')

            # Save merge MLP preds
            final_name = 'mlp_' + model_name + '_duo'
            full_oof.to_hdf(mlp_output_dir + f'{final_name}_oof.h5', key='w')
            full_test.to_hdf(mlp_output_dir + f'{final_name}_test.h5', key='w')

        # SINGLE MLP

        single_mlp_params = {
            'lr': 0.015,
            'dropout_rate': 0.3,
            'batch_size': 128,
            'num_epochs': 10000,
            'layer_dims': [200, 200, 20],
            'verbose': 0,
        }

        single_mlp_model_0 = MlpModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir=mlp_output_dir,
        )

        single_model_name = model_name + '_single'

        single_mlp_model_0.fit(params=single_mlp_params)
        single_mlp_model_0.save(single_model_name)
        # single_mlp_model_0.load('models/v10.1_single__2018-12-17_11:45:19__0.7868')

        single_mlp_model_0.predict(
            iteration_name=single_model_name,
            predict_test=True,
            save_preds=True,
            produce_sub=False,
            save_confusion=True
        )

    '''
    KNN Models
    '''
    if controls['knn-models']:
        knn_params = {
            'n_neighbors': 1,
            'weight': 'uniform',
        }

        knn_model_0 = KNNModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir='./level_1_preds/',
        )

        knn_model_0.fit(params=knn_params)

        knn_model_0.predict(
            iteration_name=model_name,
            predict_test=False,
            save_preds=True,
            produce_sub=True,
            save_confusion=True
        )

    '''
    SVM Models
    '''
    if controls['svm-models']:
        svm_params = {
            'kernel': 'rbf',
            'gamma': 0.04,
            'c': 7,
            'degree': 1,
        }

        svm_model_0 = SVCModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir='./level_1_preds/',
        )

        svm_model_0.fit(params=svm_params)

        svm_model_0.predict(
            iteration_name=model_name,
            predict_test=False,
            save_preds=True,
            produce_sub=True,
            save_confusion=True
        )

if __name__ == '__main__':
    main()