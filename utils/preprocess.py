import pandas as pd
import numpy as np
import os, pickle
from functools import reduce

def concat_feats(feat_set_list, init=None):
    '''
    Reads and concatenates already computed feats to metadata

    :param feat_set_list:
    :param init: initial dataframe containing meta feats
    :return: pandas dataframe containing all specified features with first column being object id
    '''

    feat_dfs = [pd.read_hdf(path, mode='r') for path in feat_set_list]

    def reducer(l,r):
        reduced_df = pd.merge(l,r,how='outer',on='object_id')

        return reduced_df

    if init is not None:
        return reduce(reducer, feat_dfs, init)
    else:
        return reduce(reducer, feat_dfs)

def prep_data(train_feats_list, test_feats_list, only_meta=False, augmented=False):
    '''
    Load and preprocess train and test data and metadata
    :return: tuple, as follows: (train dataframe (no target), full test dataframe,
        y_target numpy array, train_col names list)
    '''

    '''
    Metadata processing
    '''

    # Import train and test meta data
    meta_str = 'aug_' if augmented else ''
    meta_train = pd.read_csv(os.getcwd() + '/' + f'data/{meta_str}training_set_metadata.csv', dtype={'object_id': np.uint64})
    meta_test = pd.read_csv(os.getcwd() + '/' + 'data/test_set_metadata.csv')

    # Remove spectrometry redshift, distmod feats from both sets
    feats_to_delete = []
    feats_to_delete = ['hostgal_specz']
    # meta_train.drop(feats_to_delete, axis=1, inplace=True)
    # meta_test.drop(feats_to_delete, axis=1, inplace=True)

    # Create redshift bins based on meta_test data
    ng_mask = meta_test['hostgal_photoz'] > 0
    g_mask = meta_test['hostgal_photoz'] == 0

    # Number of q bins for extragalactic will be such as to have same cardinal. as galactic bin
    q = np.ceil(meta_test['hostgal_photoz'].loc[ng_mask].count() / meta_test['hostgal_photoz'].loc[g_mask].count()).astype(int)

    # Set galactic bin to -1 by default for now (will be set to zero below)
    meta_test.insert(len(meta_test.columns), 'rs_bin', -1)
    meta_train.insert(len(meta_train.columns)-1, 'rs_bin', 0) # insert bef. target col - in train galactic=0 straightway due to diff encoding process below
    # Setup extragalactic bins on meta_test
    binned_test_rs, bin_labels = pd.qcut(meta_test['hostgal_photoz'].loc[ng_mask], q, retbins=True)
    meta_test.loc[ng_mask, ['rs_bin']] = binned_test_rs.cat.codes
    meta_test['rs_bin'] += 1 # To make galactic bin 0 and move extragalactic bins up 1 cat
    # Encode train rs using test_bins
    def enc(v):
        for i,b in enumerate(bin_labels[1:]): # -1 to ignore left end of leftmost interval
            if v <= b:
                return i+1 # +1 to jump galactic bin
    meta_train.loc[meta_train['hostgal_photoz']>0, ['rs_bin']] = meta_train['hostgal_photoz'].loc[meta_train['hostgal_photoz']>0].apply(enc)

    # Get targets
    y_tgt = meta_train['target'].values
    meta_train.drop(['target'], axis=1, inplace=True)

    # Create is_galactic feat
    hg_pz = np.copy(meta_train['hostgal_photoz'].values)
    hg_pz[hg_pz > 0] = 1
    hg_pz[hg_pz == 0] = 0
    meta_train['is_galactic'] = 1 - hg_pz

    hg_pz = np.copy(meta_test['hostgal_photoz'].values)
    hg_pz[hg_pz > 0] = 1
    hg_pz[hg_pz == 0] = 0
    meta_test['is_galactic'] = 1 - hg_pz

    # Get feat col names
    train_cols = list(meta_train.columns)
    [train_cols.remove(c) for c in ['object_id', 'rs_bin', 'gal_l', 'gal_b', 'decl', 'ra', 'ddf', 'mwebv', 'distmod', 'hostgal_specz']]


    '''
    Data processing
    '''

    if not only_meta:
        train = concat_feats(train_feats_list, meta_train)
        #train.to_hdf('aug_train.h5', key='df', mode='w')
        test = concat_feats(test_feats_list, meta_test)

        # Select feat subset
        feat_subset = []
        for feat_list in train_feats_list:
            with open(feat_list.split('.h5')[0] + '.pkl', 'rb') as f:
                feat_subset.extend(pickle.load(f))

        if 'object_id' in feat_subset:
            feat_subset.remove('object_id')

        train_cols.extend(feat_subset)

    else:
        train, test = meta_train, meta_test

    return train, test, y_tgt, train_cols