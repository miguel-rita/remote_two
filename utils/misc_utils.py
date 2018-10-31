import pandas as pd
import numpy as np
import tqdm
import multiprocessing as mp
import os, pickle

def store_chunk_lightcurves_tsfresh(chunk, save_dir, save_name):
    '''
    Extract lightcurves from a chunk of data and store them on disk in tsfresh ready format

    Stores two files : a .pkl containing a list of (t, v, e) values for each lightcurve
    and a .npy 1D array containing corresponding object ids for each curve

    :param chunk: Chunk of data with at least 5 cols : object_id, passband, mjd, flux, flux_err
    :param save_dir: Full directory to save light curves and oids
    :param save_name: Name of files
    :return: --
    '''

    # Cast chunk as correct type to save memory
    dtypes = {
        'object_id': np.int32,
        'mjd': np.float32,
        'passband': np.int8,
        'flux': np.float32,
        'flux_err': np.float32,
        'detected': np.uint8,
    }
    chunk = chunk.astype(dtypes)

    df_gb = chunk.groupby(['object_id', 'passband'])
    uoids = chunk['object_id'].unique()
    passbands = range(6)

    grouped_ts = []
    for oid in tqdm.tqdm(uoids, total=uoids.size):
        for pb in passbands:
            grouped_ts.append(
                tuple(
                    df_gb.get_group((oid, pb))[series].values for series in
                    ['mjd', 'flux', 'flux_err'])
            )

    # Save oids
    np.save(save_dir + '/' + save_name + '_oids.npy', uoids)

    # Save lightcurves
    with open(save_dir + '/' + save_name + '_lcs.pkl', 'wb') as handle:
        pickle.dump(grouped_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def store_chunk_lightcurves_cesium(chunk, save_dir, save_name):
    '''
    Extract lightcurves from a chunk of data and store them on disk in cesium-ready format

    Stores two files :
        A .pkl containing a 4 element list [t, m, e, d], where:
            t, m, e, d are lists of len = # oids each, where each element x of those lists is:
                x is a len = 6 list, containing 6 np arrays with data for each passband
        The file is thus a list of lists of lists storing 4 * #oids * 6 np arrays 
    
        A .npy 1D array containing corresponding object ids of len = len(t) = len(m) = len(e) = len(d)

    :param chunk: Chunk of data with at least 6 cols : object_id, passband, mjd, flux, flux_err, detected
    :param save_dir: Full directory to save light curves and oids
    :param save_name: Name of files
    :return: --
    '''

    # Cast chunk as correct type to save memory
    dtypes = {
        'object_id': np.int32,
        'mjd': np.float64,
        'passband': np.int8,
        'flux': np.float64,
        'flux_err': np.float64,
        'detected': np.uint8,
    }
    chunk = chunk.astype(dtypes)

    df_gb = chunk.groupby(['object_id', 'passband'])
    uoids = chunk['object_id'].unique()
    passbands = range(6)

    final_list = []

    s_list = ['mjd', 'flux', 'flux_err', 'detected']
    for series in tqdm.tqdm(s_list, total=len(s_list)):

        series_oid_group = []

        for oid in uoids:

            pbs = []

            for pb in passbands:

                pbs.append(
                    df_gb.get_group((oid, pb))[series].values
                )

            series_oid_group.append(pbs)

        final_list.append(series_oid_group)

    # Save oids
    np.save(save_dir + '/' + save_name + '_oids.npy', uoids)

    # Save lightcurves
    with open(save_dir + '/' + save_name + '_lcs.pkl', 'wb') as handle:
        pickle.dump(final_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def store_chunk_lightcurves_from_path(chunk_path, save_dir, save_name):
    '''
    Wrapper for store_chunk_lightcurves, to allow for direct storing from path

    :param chunk: Path to chunk of data with at least 5 cols : object_id, passband, mjd, flux, flux_err
    :param save_dir: Full directory to save light curves and oids
    :param save_name: Name of files
    :return: --
    '''

    # Load curves
    df = pd.read_hdf(chunk_path)

    # Store curves
    store_chunk_lightcurves_cesium(chunk=df, save_dir=save_dir, save_name=save_name)

def load_lightcurves_from_path(full_dir_to_file):
    '''
    Load lightcurve data. Must provide full path and filename

    :param full_dir_to_file:
    :return:
    '''

    with open(full_dir_to_file, 'rb') as handle:
        return pickle.load(handle)

def atomic_worker(args):
    _chunks_dir, _ck_path, _save_dir = args
    store_chunk_lightcurves_from_path(
        chunk_path=_chunks_dir + '/' + _ck_path,
        save_name=_ck_path.split('.')[0],
        save_dir=_save_dir,
    )

def convert_chunks_to_lc_chunks(chunks_dir, n_batches, save_dir):
    '''
    Grab chunks in a dir and convert them to grouped lc data in another dir

    :param chunks_dir:
    :param n_batches: apply multiprocessing to one batch at a time
    :param save_dir: (str) full save dir
    :return:
    '''

    ck_paths = [p for p in os.listdir(chunks_dir) if p[0] == 't']

    atomic_args = []
    for ck_path in ck_paths:
        atomic_args.append((chunks_dir, ck_path, save_dir))

    batch_split_indexes = np.array_split(np.arange(len(atomic_args)), n_batches)

    # Dispatch work to processes, one batch at a time

    for ixs in tqdm.tqdm(batch_split_indexes, total=len(batch_split_indexes), postfix='Batch'):
        pool = mp.Pool(processes=mp.cpu_count())
        pool.map(atomic_worker, [atomic_args[ix] for ix in ixs])
        pool.close()
        pool.join()

convert_chunks_to_lc_chunks(
    chunks_dir='/Users/miguelrita/Documents/Kaggle/remote_two/data/training_chunks',
    save_dir='/Users/miguelrita/Documents/Kaggle/remote_two/data/training_cesium_curves',
    n_batches=1,
)