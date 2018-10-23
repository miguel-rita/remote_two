import pandas as pd
import numpy as np
import tqdm
import os, pickle

def store_chunk_lightcurves(chunk, save_dir, save_name):
    '''
    Extract lightcurves from a chunk of data and store them on disk

    Stores two files : a .pkl containing a list of (t, v, e) values for each lightcurve
    and a .npy 1D array containing corresponding object ids for each curve

    :param chunk: Chunk of data with at least 5 cols : object_id, passband, mjd, flux, flux_err
    :param save_dir: Local directory to save light curves and oids
    :param save_name: Name of files
    :return: --
    '''

    df_gb = chunk.groupby(['object_id', 'passband'])
    uoids = chunk['object_id'].unique()
    passbands = range(6)

    grouped_ts = []
    for oid in tqdm.tqdm(uoids, total=uoids.size):
        for pb in passbands:
            grouped_ts.append(
                tuple(
                    df_gb.get_group((oid, pb))[series].values.astype('double') for series in
                    ['mjd', 'flux', 'flux_err'])
            )

    # Save oids
    np.save(os.getcwd() + '/' + save_dir + save_name + '_oids.npy', uoids)

    # Save lightcurves
    with open(os.getcwd() + '/' + save_dir + save_name + '_lcs.pkl', 'wb') as handle:
        pickle.dump(grouped_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def store_chunk_lightcurves_from_path(chunk_path, save_dir, save_name):
    '''
    Wrapper for store_chunk_lightcurves, to allow for direct storing from path

    :param chunk: Path to chunk of data with at least 5 cols : object_id, passband, mjd, flux, flux_err
    :param save_dir: Local directory to save light curves and oids
    :param save_name: Name of files
    :return: --
    '''

    # Load curves
    df = pd.read_csv(chunk_path,
                     dtype={
                         'object_id': np.int32,
                         'mjd': np.float32,
                         'passband': np.int8,
                         'flux': np.float32,
                         'flux_err': np.float32,
                         'detected': np.uint8,
                     })

    # Store curves
    store_chunk_lightcurves(chunk=df, save_dir=save_dir, save_name=save_name)

def load_lightcurves_from_path(full_dir_to_file):
    '''
    Load lightcurve data. Must provide full path and filename

    :param full_dir_to_file:
    :return:
    '''

    with open(full_dir_to_file, 'rb') as handle:
        return pickle.load(handle)