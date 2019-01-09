import utils.preprocess as utils
from utils.preprocess import concat_feats
from lgbm import LgbmModel
from nn import MlpModel

def main():

    '''
    Load metafeats from level 2
    '''

    level_2_feats_train = [
        'level_2_preds/lgbm_v9.3_bi-covar_0.5545_oof.h5',
        'level_2_preds/mlp_mlp_v9.6_bicovar_single_0.6073_oof.h5',
        # 'level_1_preds/mlp_v9.3_single_0.8426_oof.h5',
    ]
    level_2_feats_test = [
        'level_2_preds/lgbm_v9.3_bi-covar_0.5545_test.h5',
        'level_2_preds/mlp_mlp_v9.6_bicovar_single_0.6073_test.h5',
        # 'level_1_preds/mlp_v9.3_single_0.8426_test.h5',
    ]

    train_meta, test_meta, y_tgt, selected_cols = utils.prep_data([], [], only_meta=True)

    train = concat_feats(level_2_feats_train)
    test = concat_feats(level_2_feats_test)

    selected_cols = [c for c in train.columns if c != 'object_id']

    train['hostgal_specz'] = train_meta['hostgal_specz']
    test['rs_bin'] = test_meta['rs_bin']

    train['hostgal_photoz_err'] = train_meta['hostgal_photoz_err']
    test['hostgal_photoz_err'] = test_meta['hostgal_photoz_err']
    selected_cols.append('hostgal_photoz_err')

    controls = {
        'lgbm-models': bool(1),
        'mlp-models': bool(0),
        'knn-models': bool(0),
        'svm-models': bool(0),
    }
    model_name = 'v10_doublestack'

    '''
    LGBM models
    '''
    if controls['lgbm-models']:

        fit_params = {
            'num_leaves' : 2,
            'learning_rate': 0.02,
            'min_child_samples' : 20,
            'n_estimators': 300,
            'reg_alpha': 1,
            'reg_lambda': 5,
            'bagging_fraction' : 0.8,
            'bagging_freq' : 1,
            'bagging_seed' : 1,
            'silent': -1,
            'verbose': -1,
        }

        lgbm_model_level_3 = LgbmModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir='./level_3_preds/',
            fit_params=fit_params,
        )

        '''
        Generate preds
        '''
        lgbm_model_level_3.fit_predict(
            iteration_name=model_name,
            predict_test=False,
            save_preds=True,
            produce_sub=False,
            save_imps=True,
            save_confusion=True
        )

if __name__ == '__main__':
    main()