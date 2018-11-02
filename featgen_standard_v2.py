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
    func_names = ['flux_'+n for n in ['mean', 'max', 'min', 'std', 'skew', 'kurt']]
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
    m-feats (filtered flux) - for now empty bands
    '''

    # Define feats to compute
    feats_to_compute = [np.mean, np.max, np.min, np.std, skew, kurtosis]
    num_bands = 6

    func_names = [f'is_band_{i:d}' for i in range(num_bands)]
    feat_names.extend(func_names)

    # Allocate numpy placeholder for computed feats
    m2_array = np.zeros(
        shape=(oids.size, num_bands)
    )

    # Compute 'm' (flux) feats
    for i, (oid_curves, detect_curves) in enumerate(zip(lcs[1], lcs[3])):
        for k, (band, dband) in enumerate(zip(oid_curves, detect_curves)):
            detections = dband.astype(bool)
            if not np.any(detections):
                m_array[i, k] = 0
            else:
                m_array[i, k] = 1


    '''
    td-related feats (mjd, detected)
    '''
    num_feats = 3

    # Allocate numpy placeholder for computed feats
    t_array = np.zeros(
        shape=(oids.size, num_feats)
    )

    # Define feats to compute
    def detected_feats(ts, ds):
        '''
        Compute several detected-related feats on aggregated channels:
            1) Max 1's amplitude
            2) Num. 1 blocks
            3) Avg. 1 block duration
        '''

        # Get all bands info in a single matrix
        td_matrix = np.vstack([np.hstack(ts), np.hstack(ds)]).T

        # Sort by mjd
        td_matrix = td_matrix[td_matrix[:, 0].argsort()]

        # Compute mjd groups
        prev_detected = False
        mjd_groups = []  # Final var holding all collect groups
        curr_group = []  # Temp var to collect group info in loop

        for line in td_matrix:
            mjd, detected = line[0], bool(line[1])
            if prev_detected and not detected:
                # Just finished group
                mjd_groups.append(curr_group)
                curr_group = []
                prev_detected = False
            if detected:
                # Going through/starting group
                curr_group.append(mjd)
                prev_detected = True

        # Append last group
        if curr_group:
            mjd_groups.append(curr_group)

        # Compute feats

        # Amplitude
        amp = mjd_groups[-1][-1] - mjd_groups[0][0]
        # Num. of groups (1-blocks)
        n_gps = len(mjd_groups)
        # Average group duration
        mean_gp = np.mean([g[-1] - g[0] for g in mjd_groups])

        return np.array([
            amp,
            n_gps,
            mean_gp,
        ])


    feats_to_compute = [detected_feats]

    feat_names.extend([
        'det_amplitude',
        'det_n_1_blocks',
        'det_avg_1_block_duration',

    ])


    # Compute 'td' (mjd/detected)-related feats
    for i,(oid_t_curves, oid_d_curves) in enumerate(zip(lcs[0], lcs[3])):
        for j,f in enumerate(feats_to_compute):
            t_array[i,:] = f(oid_t_curves, oid_d_curves)




    '''
    Simple d feats
    '''

    # Define feats to compute
    feats_to_compute = [np.mean, np.std, skew]
    num_bands = 6
    func_names = ['detected_'+n for n in ['mean', 'std', 'skew']]

    for fn in func_names:
        feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

    # Allocate numpy placeholder for computed feats
    d_array = np.zeros(
        shape=(oids.size, num_bands * len(func_names))
    )

    # Compute 'd' (detected) feats
    for i, oid_curves in enumerate(lcs[3]):
        for j, f in enumerate(feats_to_compute):
            for k, band in enumerate(oid_curves):
                d_array[i, j * num_bands + k] = f(band)





    '''
    Agg feats and wrap up
    '''

    # Stack oids to feat results
    df_array = np.hstack([np.expand_dims(oids, 1), m_array, t_array, d_array, m2_array])

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
    for lcs_path, oids_path in zip(sorted(glob.glob(light_curves_dir + '/*.pkl')), sorted(glob.glob(light_curves_dir + '/*.npy'))):
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
    save_name=set_str+'_set_feats_v3_isband.h5',
    light_curves_dir='data/'+set_str+'_cesium_curves',
    n_batches=1,
)
print(f'>   featgen_standard_v2 : Wall time : {(time.time()-st):.2f} seconds')

# Featgen test
# print(detected_feats(
#     ts=[
#         [0,1,2,7,8,9],
#         [4,5,6,100,105,106,200],
#     ],
#     ds=[
#         [0,1,1,1,1,0],
#         [0,1,1,0,1,1,1],
#     ],
# ))
