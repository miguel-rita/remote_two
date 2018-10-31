'''
Generate standard (custom) feats from saved and already grouped lc data in cesium-ready
format
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, glob, time, gc, pickle
from scipy.stats import kurtosis, skew

def atomic_worker(args):

    # Load grouped curve info
    lcs_dir, oids_dir = args
    with open(lcs_dir, 'rb') as f:
        lcs = pickle.load(f)
    oids = np.load(oids_dir).astype(np.uint32)

    '''
    m-feats (flux)
    '''

    # Define feats to compute
    feats_to_compute = [np.mean, np.max, np.min, np.std, skew, kurtosis]
    num_bands = 6
    func_names = ['mean', 'max', 'min', 'std', 'skew', 'kurt']
    feat_names = []
    for fn in func_names:
        feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

    # Allocate numpy placeholder for computed feats
    m_array = np.zeros(
        shape=(oids.size, len(feat_names))
    )

    # Compute 'm' (flux) feats
    for i,oid_curves in enumerate(lcs[1]):
        for j,f in enumerate(feats_to_compute):
            for k,band in enumerate(oid_curves):
                m_array[i,j*num_bands+k] = f(band)

    '''
    t-related feats (mjd)
    '''

    # Define feats to compute
    def detect_amplitude(ts, ds):
        '''
        Compute max detected span across passbands 
        '''
        detected_stamps = []
        for t, d in zip(ts, ds):
            d = d.astype(np.bool)
            if np.any(d):
                t = t[d]
                detected_stamps.extend([t[0], t[-1]])
        return np.max(detected_stamps) - np.min(detected_stamps)


    feats_to_compute = [detect_amplitude]

    feat_names.extend(['detected_amplitude'])

    # Allocate numpy placeholder for computed feats
    t_array = np.zeros(
        shape=(oids.size, 1)
    )

    # Compute 't' (mjd)-related feats
    for i,(oid_t_curves, oid_d_curves) in enumerate(zip(lcs[0], lcs[3])):
        for j,f in enumerate(feats_to_compute):
            t_array[i,0] = f(oid_t_curves, oid_d_curves)

    '''
    Agg feats and wrap up
    '''

    # Stack oids to feat results
    df_array = np.hstack([np.expand_dims(oids, 1), m_array, t_array])

    # Build final pandas dataframe
    df = pd.DataFrame(
        data=df_array,
        columns=['object_id'] + feat_names,
    )

    del lcs
    gc.collect()

    return df


def main(save_dir, save_name, light_curves_dir, n_batches):
    '''
    Generate custom features data frame from stored grouped light curve data

    :param save_dir (str) Dir to save calculated feats
    :param save_name (str) Feat set name
    :param light_curves_dir (str) 
    :param n_batches (int) Process using multiprocess on one batch of saved lc data at a time
    :return:
    '''

    np.warnings.filterwarnings('ignore')


    # Get paths to lcs and respective oids
    atomic_args = []
    for lcs_path, oids_path in zip(glob.glob(light_curves_dir + '/*.pkl'), glob.glob(light_curves_dir + '/*.npy')):
        atomic_args.append((lcs_path, oids_path))


    # Dispatch work to processes, one batch at a time
    batch_split_indexes = np.array_split(np.arange(len(atomic_args)), n_batches)

    print(f'>   featgen_standard_v2 : Starting batch multiprocessing . . .')
    res = []
    for ixs in tqdm.tqdm(batch_split_indexes, total=len(batch_split_indexes), postfix='Batch'):
        pool = mp.Pool(processes=mp.cpu_count())
        res.extend(pool.map(atomic_worker, [atomic_args[ix] for ix in ixs]))
        pool.close()
        pool.join()

    print(f'>   featgen_standard_v2 : Concating and saving results . . .')

    # Concat atomic computed feats, sort by oid, reset types, ready to test
    df = pd.concat(res, axis=0).sort_values('object_id')

    types_dict = {feat_name_: np.float32 for feat_name_ in df.columns[1:]}
    types_dict['object_id'] = np.uint32
    df = df.astype(types_dict)

    df.reset_index(drop=True).to_hdf(save_dir+'/'+save_name, key='w')


set_str = 'training'
st = time.time()
main(
    save_dir='data/'+set_str+'_feats',
    save_name=set_str+'_set_feats_first_from_grouped_plus_detected_one.h5',
    light_curves_dir='data/'+set_str+'_cesium_curves',
    n_batches=1,
)
print(f'>   featgen_standard_v2 : Wall time : {(time.time()-st):.2f} seconds')

