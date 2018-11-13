import utils.preprocess as utils
from lgbm import LgbmModel
from nn import MlpModel

def main():

    '''
    Load and preprocess data
    '''

    train_feats_list = [
        'data/training_feats/training_set_feats_r3_m-feats_v3.h5',
        'data/training_feats/training_set_feats_r3_t-feats_v1.h5',
        'data/training_feats/training_set_feats_r3_d-feats_v1.h5',
        'data/training_feats/training_set_feats_r3_linreg-feats_v4.h5',
    ]
    test_feats_list = [
        'data/test_feats/test_set_feats_r3_m-feats_v3.h5',
        'data/test_feats/test_set_feats_r3_t-feats_v1.h5',
        'data/test_feats/test_set_feats_r3_d-feats_v1.h5',
        'data/test_feats/test_set_feats_r3_linreg-feats_v4.h5',
    ]
    train, test, y_tgt, selected_cols = utils.prep_data(train_feats_list, test_feats_list)

    controls = {
        'lgbm-models'   : bool(1),
        'mlp-models'    : bool(0),
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
            test=test,
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

if __name__ == '__main__':
    main()