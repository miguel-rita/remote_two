import utils.preprocess as utils
from utils.preprocess import concat_feats
from lgbm import LgbmModel

def main():

    '''
    Load metafeats from level 1
    '''

    level_1_feats_train = [
        'level_1_preds/lgbm_v5.0_0.5706_oof.h5',
        'level_1_preds/mlp_v5.0_0.8384_oof.h5',
    ]
    level_1_feats_test = [
        'level_1_preds/lgbm_v5.0_0.5706_test.h5',
        'level_1_preds/mlp_v5.0_0.8384_test.h5',
    ]

    train, test, y_tgt, selected_cols = utils.prep_data([], [], only_meta=True)

    train = concat_feats(level_1_feats_train)
    test = concat_feats(level_1_feats_test)

    selected_cols = [c for c in train.columns if c != 'object_id']

    '''
    LGBM models
    '''

    fit_params = {
        'num_leaves': 2,
        'learning_rate': 0.05,
        'n_estimators': 10000,
        'reg_alpha': 1,
        'reg_lambda': 5,
        'silent': True,
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
        iteration_name='v5.0_lvl2_new_peaks15mean',
        predict_test=False,
        save_preds=True,
        produce_sub=True,
        save_imps=True,
        save_confusion=True
    )

if __name__ == '__main__':
    main()