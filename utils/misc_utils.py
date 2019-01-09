import pandas as pd
import numpy as np
import tqdm
import multiprocessing as mp
import os, pickle, gc, glob
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def plot_confusion_matrix(cm, classes, filename_, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    nclasses = cm.shape[0]

    plt.figure(figsize=(10 * nclasses / 14, 8 * nclasses / 14))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{filename_}.png')
    plt.clf()

def save_importances(imps_, filename_):
    mean_gain = imps_[['gain', 'feat']].groupby('feat').mean().reset_index()
    mean_gain.index.name = 'feat'
    plt.figure(figsize=(6, 17))
    sns.barplot(x='gain', y='feat', data=mean_gain.sort_values('gain', ascending=False))
    plt.title(f'Num. feats = {mean_gain.shape[0]:d}')
    plt.tight_layout()
    plt.savefig(filename_+'.png')
    plt.clf()

def save_submission(y_test, sub_name, rs_bins, nrows=None):

    # Get submission header
    col_names = list(pd.read_csv(filepath_or_buffer='data/sample_submission.csv', nrows=1).columns)
    num_classes = len(col_names) - 1

    # Get test ids
    object_ids = pd.read_csv(filepath_or_buffer='data/test_set_metadata.csv', nrows=nrows,
                             usecols=['object_id']).values.astype(int)
    num_ids = object_ids.size

    # Class 99 ref prob
    reference_prob = 0.16

    c99_cols = 1 - y_test
    num = np.multiply.reduce(c99_cols, axis=1, keepdims=True)
    den = np.mean(c99_cols, axis=1, keepdims=True)
    c99_col = num/den
    avg_prob = np.mean(c99_col)
    equalizing_const = reference_prob / avg_prob
    c99_col *= equalizing_const

    y_test *= (1 - c99_col)
    sub = np.hstack([object_ids, y_test, c99_col])

    h = ''
    for s in col_names:
        h += s + ','
    h = h[:-1]

    # Write to file
    np.savetxt(
        fname=sub_name,
        X=sub,
        fmt=['%d'] + ['%.3f'] * num_classes,
        delimiter=',',
        header=h,
        comments='',
    )

def mask_samples(tgt, mask_dict):
    '''
    Compute a boolean mask to cover tgt classes as specified by mask_dict.
    Mask_dict entries as follows:
    k : v, where k is the tgt class to mask, and v is a (v0, v1) tuple, where v0 is the number of splits to
    make in class k, and v1 is the split number to leave unmasked (ie. True). All other splits are masked
    '''

    # Full mask
    mask = np.ones(tgt.size)

    for k, v in mask_dict.items():
        # Get tgt indices
        k_ixs = np.nonzero(tgt == k)[0]
        # Split tgt indices
        split_k_ixs = np.array_split(k_ixs, v[0])

        # Cover all k splits except the choosen split
        for i, s in enumerate(split_k_ixs):
            if i != v[1]:
                mask[s] = 0

    return mask.astype(bool)

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

    del chunk, df_gb, uoids, passbands, final_list
    gc.collect()

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

    del df
    gc.collect()

def store_metadata_by_chunks(curves_dir, dataset):
    '''
    Store metadata corresponding to each lcs/oids pair in 'curves_dir', in .h5 format
    
    :param curves_dir: Directory to curves and oids in cesium-like format
    :param dataset: (str) training or test
    :return: --
    '''

    # Load metadata
    meta = pd.read_csv(f'../data/{dataset}_set_metadata.csv')

    oids_paths = glob.glob(curves_dir + '/*.npy')

    # For each chunk save corresponding meta part
    for oids_path in tqdm.tqdm(oids_paths, total=len(oids_paths)):
        oids = np.load(oids_path)
        meta_name = oids_path.split('oids')[0] + 'meta.h5'
        chunk_meta = meta[np.isin(meta['object_id'].values, oids)]
        chunk_meta.to_hdf(meta_name, key='meta_df', mode='w')

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
        pool = mp.Pool(processes=4)
        pool.map(atomic_worker, [atomic_args[ix] for ix in ixs])
        pool.close()
        pool.join()

# os.chdir('..')
# set_name = 'training'
# convert_chunks_to_lc_chunks(
#     chunks_dir=os.getcwd() + f'/data/{set_name}_chunks',
#     save_dir=os.getcwd() + f'/data/{set_name}_cesium_curves',
#     n_batches=9,
# )
