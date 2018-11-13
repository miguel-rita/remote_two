import utils.preprocess as utils
from lgbm import LgbmModel
from nn import MlpModel
from rf import RfModel

def main():

    '''
    Load and preprocess data
    '''

    train_feats_list = [
        'data/training_feats/training_set_feats_r4_m-feats_v1.h5',
        'data/training_feats/training_set_feats_r4_t-feats_v1.h5',
        'data/training_feats/training_set_feats_r4_d-feats_v1.h5',
        'data/training_feats/training_set_feats_r4_peak-feats_v1.h5',
        #'data/training_feats/training_set_feats_r4_peak-feats_v2_30maxmean.h5',
        'data/training_feats/training_set_feats_r4_linreg-feats_v1.h5',
    ]
    test_feats_list = [
        'data/test_feats/test_set_feats_std.h5',
        # 'data/test_feats/training_set_feats_r4_m-feats_v1.h5',
        # 'data/test_feats/training_set_feats_r4_t-feats_v1.h5',
        # 'data/test_feats/training_set_feats_r4_d-feats_v1.h5',
        # 'data/test_feats/training_set_feats_r4_linreg-feats_v1.h5',
    ]
    train, test, y_tgt, selected_cols = utils.prep_data(train_feats_list, test_feats_list)

    controls = {
        'lgbm-models'   : bool(0),
        'mlp-models'    : bool(0),
        'rf-models'     : bool(1),
    }

    '''
    LGBM Models
    '''

    if controls['lgbm-models']:
        lgbm_params = {
            'num_leaves' : 7,
            'learning_rate': 0.10,
            'n_estimators': 10000,
            'reg_alpha': 1,
            'reg_lambda': 5,
            'silent': True,
        }

        lgbm_model_0 = LgbmModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir='./level_1_preds/',
            fit_params=lgbm_params
        )

        model_name = 'v4.5'

        lgbm_model_0.fit_predict(
            iteration_name=model_name,
            predict_test=False,
            save_preds=True,
            produce_sub=False,
            save_imps=True,
            save_confusion=True
        )

    '''
    MLP models
    '''

    if controls['mlp-models']:
        mlp_params = {
            'lr': 0.01,
            'dropout_rate': 0.25,
            'batch_size': 256,
            'num_epochs': 10000,
            'layer_dims': [200, 100],
        }

        mlp_model_0 = MlpModel(
            train=train,
            test=None,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir='./level_1_preds/',
        )

        model_name = 'v4.5'

        mlp_model_0.fit(params=mlp_params)
        mlp_model_0.save(model_name)
        # mlp_model_0.load('models/v4.4__2018-11-13_11:40:04__X')

        '''
        Predict using mlp models
        '''
        mlp_model_0.predict(
            iteration_name=model_name,
            predict_test=False,
            save_preds=True,
            produce_sub=False,
            save_confusion=True
        )

    '''
    RF Models
    '''

    if controls['rf-models']:
        rf_params = {
            'n_estimators'  : 100,
            'max_depth'     : None,
            'max_features'  : None,
        }

        rf_model_0 = RfModel(
            train=train,
            test=None,
            y_tgt=y_tgt,
            selected_cols=selected_cols,
            output_dir='./level_1_preds/',
            params=rf_params
        )

        model_name = 'v4.5'

        rf_model_0.fit_predict(
            iteration_name=model_name,
            predict_test=False,
            save_preds=True,
            produce_sub=False,
            save_imps=True,
            save_confusion=True
        )

if __name__ == '__main__':
    main()