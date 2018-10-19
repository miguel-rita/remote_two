import pandas as pd
import numpy as np
import os, time, tqdm
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

def prep_data():
    '''
    Load and preprocess train and test data
    :return: tuple, as follows: (train dataframe (no target), full test dataframe,
        y_target numpy array, train_col names list)
    '''

    # Import train and test meta data
    train = pd.read_csv(os.getcwd() + '/' + 'data/training_set_metadata.csv')
    test = pd.read_csv(os.getcwd() + '/' + 'data/test_set_metadata.csv')

    # Check overall info
    print(train.columns, train.shape)
    print(test.columns, test.shape)

    # Check for NAs in metadata
    print('Num. of NAs:')
    print(train.isna().sum())
    print(test.isna().sum())

    # Remove spectrometry redshift, distmod feats from both sets
    feats_to_delete = ['hostgal_specz', 'distmod']
    train.drop(feats_to_delete, axis=1, inplace=True)
    test.drop(feats_to_delete, axis=1, inplace=True)

    # Create redshift bins based on TEST data
    ng_mask = test['hostgal_photoz'] > 0
    g_mask = test['hostgal_photoz'] == 0

    # Number of q bins for extragalactic will be such as to have same cardinal. as galactic bin
    q = np.ceil(test['hostgal_photoz'].loc[ng_mask].count() / test['hostgal_photoz'].loc[g_mask].count()).astype(int)

    # Set galactic bin to -1 by default for now (will be set to zero below)
    test.insert(len(test.columns), 'rs_bin', -1)
    train.insert(len(train.columns)-1, 'rs_bin', 0) # insert bef. target col - in train galactic=0 straightway due to diff encoding process below
    # Setup extragalactic bins on test
    binned_test_rs, bin_labels = pd.qcut(test['hostgal_photoz'].loc[ng_mask], q, retbins=True)
    test.loc[ng_mask, ['rs_bin']] = binned_test_rs.cat.codes
    test['rs_bin'] += 1 # To make galactic bin 0 and move extragalactic bins up 1 cat
    # Encode train rs using test_bins
    def enc(v):
        for i,b in enumerate(bin_labels[1:]): # -1 to ignore left end of leftmost interval
            if v <= b:
                return i+1 # +1 to jump galactic bin
    train.loc[train['hostgal_photoz']>0, ['rs_bin']] = train['hostgal_photoz'].loc[train['hostgal_photoz']>0].apply(enc)

    # Check feat was created
    print(train.columns, train.shape)
    print(test.columns, test.shape)

    # Check class freqs.
    print(train.groupby('target')['object_id'].count())

    # Get targets
    y_tgt = train['target'].values
    train.drop(['target'], axis=1, inplace=True)

    # Get feat col names
    train_cols = list(train.columns)
    [train_cols.remove(c) for c in ['object_id', 'rs_bin']]

    return (train, test, y_tgt, train_cols)