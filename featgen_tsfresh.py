import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tsfresh.utilities.dataframe_functions import impute
import os, time
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

df = pd.read_csv(
    'data/training_set.csv',
    dtype =
    {
        'object_id': np.int32,
        'mjd': np.float32,
        'passband': np.int8,
        'flux': np.float32,
        'flux_err': np.float32,
        'detected': np.uint8,
    },
)

params = {
    'abs_energy' : None,
    'cid_ce' : [{'normalize' : False}],
    'count_above_mean' : None,
    'count_below_mean' : None,
    'fft_aggregated' : [
        {
            'aggtype' : 'centroid',
            'aggtype' : 'variance',
        }
    ],
    'kurtosis' : None,
    'longest_strike_below_mean' : None,
    'maximum' : None,
    'mean' : None,
    'mean_abs_change' : None,
    'median' : None,
    'minimum' : None, # Detail
    'skewness' : None,
    'standard_deviation' : None,
    'symmetry_looking' : [
        {
            'r' : 0.5,
        }
    ],
}

feats_raw = extract_features(
    timeseries_container=df,
    column_id='object_id',
    column_sort='mjd',
    column_kind='passband',
    column_value='flux',
    impute_function=impute,
    default_fc_parameters=params,
)

feats_raw.insert(
    loc=0,
    column='object_id',
    value=df['object_id'].unique()
)

feats_raw.to_hdf('data/training_feats/train_tsfresh_baseline.h5', 'w')

# relevant_feats = set()
# for label in np.unique(target):
#     target_binary = target == label
#     feats_filtered_sub = select_features(feats_raw, target_binary, ml_task='classification')
#     relevant_feats = relevant_feats.union(set(feats_filtered_sub.columns))
#
# feats_filtered = feats_raw[list(relevant_feats)]

print('Done')