import utils.preprocess as utils
from utils.preprocess import concat_feats
from lgbm import LgbmModel
from nn import MlpModel

def main():

    '''
    Load metafeats from level 1
    '''

    level_1_feats_train = [
        'level_1_preds/lgbm_v9.7_0.5688_oof.h5',
        'level_1_preds/lgbm_v9.3_0.5704_oof.h5',
        'level_1_preds/mlp_vFFF_single_0.7948_oof.h5',
        'level_1_preds/mlp_vFFF_single_0.7982_oof.h5',
        'level_1_preds/mlp_vFFF_single_0.8040_oof.h5',
        'level_1_preds/mlp_vFFF_single_0.8237_oof.h5',
        # 'level_1_preds/lgbm_v9.7_2_0.5647_oof.h5',
        # 'level_1_preds/mlp_v9.7_1_single_0.7918_oof.h5',
        'level_1_preds/mlp_v7.0.6_0.8422_oof.h5',
        'level_1_preds/mlp_v9.3_single_0.8426_oof.h5',
        'level_1_preds/mlp_v4.1_0.8900_oof.h5',
        'level_1_preds/mlp_v8.28_galactic_0.3097_oof.h5',
        'level_1_preds/mlp_v8.28_extra_1.0634_oof.h5',
    ]
    level_1_feats_test = [
        'level_1_preds/lgbm_v9.7_0.5688_test.h5',
        'level_1_preds/lgbm_v9.3_0.5704_test.h5',
        'level_1_preds/mlp_vFFF_single_0.7948_test.h5',
        'level_1_preds/mlp_vFFF_single_0.7982_test.h5',
        'level_1_preds/mlp_vFFF_single_0.8040_test.h5',
        'level_1_preds/mlp_vFFF_single_0.8237_test.h5',
        # 'level_1_preds/lgbm_v9.7_2_0.5647_test.h5',
        # 'level_1_preds/mlp_v9.7_1_single_0.7918_test.h5',
        'level_1_preds/mlp_v7.0.6_0.8422_test.h5',
        'level_1_preds/mlp_v9.3_single_0.8426_test.h5',
        'level_1_preds/mlp_v4.1_0.8900_test.h5',
        'level_1_preds/mlp_v8.28_galactic_0.3097_test.h5',
        'level_1_preds/mlp_v8.28_extra_1.0634_test.h5',
    ]

    train_meta, test_meta, y_tgt, selected_cols = utils.prep_data([], [], only_meta=True)

    train = concat_feats(level_1_feats_train)
    test = concat_feats(level_1_feats_test)

    selected_cols = [c for c in train.columns if c != 'object_id']

    # for cls in [6,92,62,90,16,52,67,15]:
    #     selected_cols.remove(f'mlp_v9.3_single_0.8426__{cls:d}')

    train['hostgal_specz'] = train_meta['hostgal_specz']
    test['rs_bin'] = test_meta['rs_bin']

    controls = {
        'lgbm-models': bool(1),
        'mlp-models': bool(0),
        'knn-models': bool(0),
        'svm-models': bool(0),
    }
    model_name = 'vFFF_Stack'

    '''
    LGBM models
    '''

    if controls['lgbm-models']:

        fit_params = {
            'num_leaves' : 2,
            'learning_rate': 0.10,
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

        lgbm_model_level_2 = LgbmModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir='./level_2_preds/',
            fit_params=fit_params,
        )

        '''
        Generate preds
        '''
        lgbm_model_level_2.fit_predict(
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

        mlp_output_dir = './level_2_preds/'

        # SINGLE MLP

        single_mlp_params = {
            'lr': 0.025,
            'dropout_rate': 0.3,
            'batch_size': 256,
            'num_epochs': 10000,
            'layer_dims': [50],
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

        # single_mlp_model_0.fit(params=single_mlp_params)
        # single_mlp_model_0.save(single_model_name)
        single_mlp_model_0.load('models/v9.6_bicovar_single__2018-12-15_16:54:31__0.6073')

        single_mlp_model_0.predict(
            iteration_name=single_model_name,
            predict_test=True,
            save_preds=True,
            produce_sub=True,
            save_confusion=True
        )


if __name__ == '__main__':
    main()