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
    [train_cols.remove(c) for c in ['object_id']]

    return (train, test, y_tgt, train_cols)