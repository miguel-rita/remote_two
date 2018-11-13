'''
Generate standard (custom) feats from saved and already grouped lc data in cesium-ready
format
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, glob, time, gc, pickle
from scipy.stats import kurtosis, skew
import cesium.time_series, cesium.featurize

def atomic_worker(args):

    # Load grouped curve info
    lcs_dir, oids_dir = args
    with open(lcs_dir, 'rb') as f:
        lcs = pickle.load(f)
    oids = np.load(oids_dir).astype(np.uint32)

    # Feat computation controls
    feat_names = []
    feat_arrays = []
    compute_feats = {
        'm-feats'            : bool(0),
        't-feats'            : bool(0),
        'd-feats'            : bool(0),
        'e-feats'            : bool(0),
        'cesium-feats'       : bool(1),
        'slope-feats'        : bool(0),
        'curve-feats'        : bool(0),
        'linreg-feats'       : bool(0),
    }

    '''
    m-feats (flux)
    '''
    if compute_feats['m-feats']:

        # Define feats to compute
        feats_to_compute = [np.average, np.max, np.min, np.std, skew, kurtosis]
        num_bands = 6
        func_names = ['flux_'+n for n in ['mean', 'max', 'min', 'std', 'skew', 'kurt']]
        for fn in func_names:
            feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

        # Allocate numpy placeholder for computed feats
        m_array = np.zeros(
            shape=(oids.size, len(feat_names))
        )

        # Compute 'm' (flux) feats
        for i,(flux_curves, err_curves, dect_curves) in enumerate(zip(lcs[1], lcs[2], lcs[3])):
            for j,f in enumerate(feats_to_compute):

                for k,(flux_curve, err_curve, dect_curve) in enumerate(zip(flux_curves, err_curves, dect_curves)):

                    if f == np.average:
                        res = f(flux_curve, weights=1/err_curve)
                    else:
                        res = f(flux_curve)

                    m_array[i, j * num_bands + k] = res

        feat_arrays.append(m_array)

        # Compute cross flux relations
        cross_band_names = ['cross_band_flux_mean_contrib', 'cross_band_flux_max_contrib']
        for fn in cross_band_names:
            feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

        # # For bands 2-4
        # cross_band_names = ['red_cross_band_flux_mean_contrib', 'red_cross_band_flux_max_contrib']
        # for fn in cross_band_names:
        #     feat_names.extend([f'{fn}_{i:d}' for i in [2,3,4]])

        mc_array = np.hstack([
            m_array[:,:6] / np.sum(m_array[:,:6], axis=1)[:,None],
            m_array[:,6:12] / np.sum(m_array[:,6:12], axis=1)[:,None],
            # m_array[:,2:5] / np.sum(m_array[:,2:5], axis=1)[:,None],
            # m_array[:,8:11] / np.sum(m_array[:,8:11], axis=1)[:,None],
        ])
        feat_arrays.append(mc_array)

    '''
    t-related feats (mjd, detected)
    '''
    if compute_feats['t-feats']:

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

        feat_arrays.append(t_array)

    '''
    d-feats
    '''
    if compute_feats['d-feats']:

        # Define feats to compute
        feats_to_compute = [np.mean]
        num_bands = 6
        func_names = ['detected_'+n for n in ['mean']]
        local_names = ['detected_'+n for n in ['mean']]

        for fn in local_names:
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

        # d2_array = np.max(d_array[:,6:], axis=1, keepdims=True)
        # for i,e in enumerate(d2_array):
        #     if e <= 4:
        #         d2_array[i] = 10000
        #     else:
        #         d2_array[i] = 0
        # feat_names.extend(['detected_maxsum'])

        feat_arrays.extend([d_array])

    '''
    e-feats (filtered flux_err)
    '''
    if compute_feats['e-feats']:

        # Define feats to compute
        feats_to_compute = [np.mean, np.max, np.std]
        num_bands = 6

        func_names = ['ferr_' + n for n in ['mean', 'max', 'std']]
        local_names = []
        for fn in func_names:
            local_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

        #feat_names.extend(local_names)

        # Allocate numpy placeholder for computed feats
        e_array = np.zeros(
            shape=(oids.size, len(local_names))
        )

        # Compute 'm' (flux) feats
        for i, (err_curves, detect_curves) in enumerate(zip(lcs[2], lcs[3])):
            for j,f in enumerate(feats_to_compute):
                for k,(err_curve, detect_curve) in enumerate(zip(err_curves, detect_curves)):

                    detect_curve = detect_curve.astype(np.bool)

                    if not np.any(detect_curve):
                        e_array[i, j * num_bands + k] = 0
                    else:
                        e_array[i,j*num_bands+k] = f(err_curve[detect_curve])

        #feat_arrays.append(e_array)

        # Compute cross flux relations
        cross_band_names = ['cross_band_ferr_mean_contrib', 'cross_band_ferr_max_contrib']
        for fn in cross_band_names:
            feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])
        ec_array = np.hstack([
            e_array[:, :6] / np.sum(e_array[:, 6:12], axis=1)[:, None],
            e_array[:, 6:12] / np.sum(e_array[:, 6:12], axis=1)[:, None],
        ])
        feat_arrays.append(ec_array)

    '''
    Cesium feats
    '''
    if compute_feats['cesium-feats']:

        # Define feats to compute
        cesium_feat_names = [
            'qso_log_chi2_qsonu',
            'qso_log_chi2nuNULL_chi2nu',
            # 'stetson_j',
            # 'stetson_k',
            # 'amplitude',
            # 'max_slope',
            # 'median_absolute_deviation',
        ]

        num_bands = 6

        for fn in cesium_feat_names:
            feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

        # Allocate numpy placeholder for computed feats
        c_array = np.zeros(
            shape=(oids.size, num_bands * len(cesium_feat_names))
        )

        # Compute cesium feats
        for i, (t_curve, m_curve, e_curve) in tqdm.tqdm(enumerate(zip(lcs[0], lcs[1], lcs[2])), total=len(lcs[0])):
            ts = cesium.time_series.TimeSeries(t=t_curve, m=m_curve, e=e_curve)
            dict_feats = cesium.featurize.featurize_single_ts(ts, cesium_feat_names)
            for j, ces_feat in enumerate(cesium_feat_names):
                c_array[i,j:j+num_bands] = dict_feats[ces_feat].values

        feat_arrays.append(c_array)

    '''
    slope feats
    '''
    if compute_feats['slope-feats']:

        def slope_feats(ts, ms, ds):
            '''
            Custom compute of slope feats
            '''

            mask = (ds[1:] * ds[:-1]) # Mask to compute slope between detected pts only
            mask = mask.astype(np.bool)

            if np.sum(ds) <= 1: # 0 or 1 detected points are insufficient for slope computation
                return np.zeros(3)

            s = (ms[1:]-ms[:-1])/(ts[1:]-ts[:-1])


            return np.array([np.mean(s), np.std(s), skew(s)])

        # Define feats to compute
        feats_to_compute = [slope_feats]
        num_bands = 6
        local_names = []
        for fn in ['slope_mean', 'slope_std', 'slope_skew']:
            local_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

        feat_names.extend(local_names)

        # Allocate numpy placeholder for computed feats
        s_array = np.zeros(
            shape=(oids.size, len(local_names))
        )

        # Compute slope feats
        for i, (t_curves, m_curves, d_curves) in enumerate(zip(lcs[0], lcs[1], lcs[3])):
            for j, (ts, ms, ds) in enumerate(zip(t_curves, m_curves, d_curves)):
                for f in feats_to_compute:
                    fts = f(ts, ms, ds)
                    _nfeats = len(fts)
                    s_array[i, j*_nfeats:j*_nfeats+_nfeats] = fts

        feat_arrays.append(s_array)

    '''
    Curve fitting feats
    '''
    if compute_feats['curve-feats']:

        cn = [42,52,62,67,90]
        num_feats = len(cn) # 5 sn curve models

        # Allocate numpy placeholder for computed feats
        c_array = np.zeros(
            shape=(oids.size, num_feats)
        )

        local_feat_names = [f'sn_{cn_}' for cn_ in cn]
        # for j in [2,3,4]:
        #     feat_names.extend([f'{fn}_{j:d}' for fn in local_feat_names])
        feat_names.extend(local_feat_names)

        # Load sn models
        with open('data/sn_models.pickle', 'rb') as handle:
            sn_models = pickle.load(handle)
        MAX_X = 150
        MIN_X = -50
        n_divs = MAX_X - MIN_X + 1
        standard_time_stamps = np.linspace(MIN_X, MAX_X, n_divs)
        WEIGHTS = np.ones(n_divs)
        WEIGHTS[:50] = 0
        WEIGHTS[75:] = 0.25

        band_fit_array = np.zeros(
            shape=(oids.size, num_feats * 3) #2 models, 3 bands
        )

        # Compute flux curve fit for bands 2,3,4 for each of the 5 models
        # In the end 5 feats, representing overall curve fit to each of the 5 models
        for i, (t_curves, m_curves, e_curves, d_curves) in enumerate(zip(lcs[0], lcs[1], lcs[2], lcs[3])):
            for j, (t_band, m_band, e_band, d_band) in enumerate(zip(t_curves, m_curves, e_curves, d_curves)):

                if j<2 or j==5:
                    continue

                adjusted_j = j - 2  # Since we started on band 2

                # Filter series
                if np.sum(d_band) > 1: # One point also excluded since it would have perfect fit

                    # Filtered only
                    m_band = m_band[d_band.astype(bool)]
                    t_band = t_band[d_band.astype(bool)]
                    e_band = e_band[d_band.astype(bool)]

                    # Identify peak mjd and transpose series to peak at t=50
                    t_band -= t_band[np.argmax(m_band)]

                    # Normalize series
                    m_band /= np.max(m_band)

                    # Interpolate series
                    m_band = np.interp(standard_time_stamps, t_band, m_band, left=-1, right=-1)
                    e_band = np.interp(standard_time_stamps, t_band, e_band, left=-1, right=-1)

                    # For each sn model for this band(2,3,4)
                    for k, mod_num in enumerate(cn):

                        # Load band model
                        band_model = sn_models[mod_num][j]

                        # Compute mean distance to model weighted by error and model weights
                        mask = m_band != -1 # Pick interpolated pts, not extrapolated pts

                        mse = (band_model[mask] - m_band[mask])**2 * WEIGHTS[mask] / e_band[mask]

                        band_fit_array[i, adjusted_j*num_feats + k] = np.mean(mse)

                else:
                    # If band non-existant fit_dist = -1 to all models (used later to exclude from global fit score)
                    band_fit_array[i, adjusted_j*num_feats:adjusted_j*num_feats+num_feats] = -1

        # Aggregate scores across bands
        for feat_num in range(num_feats):
            for jjj, line in enumerate(band_fit_array):
                model_scores = line[[feat_num, feat_num+num_feats, feat_num+num_feats*2]]
                # Leave -1 out of mean
                if np.any(model_scores!=-1):
                    c_array[jjj,feat_num] = np.mean(model_scores[model_scores!=-1])
                else:
                    c_array[jjj, feat_num] = 100

        feat_arrays.append(c_array)

    '''
    Linreg feats
    '''
    if compute_feats['linreg-feats']:

        linreg_feats = ['back', 'front']
        lin_feats = ['mean']#, 'std']
        num_feats = len(linreg_feats) # Back and front linreg fits

        # Allocate numpy placeholder for computed feats
        linreg_array = np.zeros(
            shape=(oids.size, num_feats * len(lin_feats))
        )


        local_feat_names = [f'linreg_b1_{cn_}' for cn_ in linreg_feats]
        for j in lin_feats:
            feat_names.extend([f'{fn}_{j}' for fn in local_feat_names])

        # Band info array for later collapse
        linreg_bands = np.zeros(
            shape=(oids.size, num_feats * 6)  # six bands per feat
        )

        # Compute simple b1 from linreg from normalized peak
        for i, (t_curves, m_curves, e_curves, d_curves) in enumerate(zip(lcs[0], lcs[1], lcs[2], lcs[3])):
            for j, (t_band, m_band, e_band, d_band) in enumerate(zip(t_curves, m_curves, e_curves, d_curves)):

                # Filtered only
                m_band = m_band[d_band.astype(bool)]
                t_band = t_band[d_band.astype(bool)]
                e_band = e_band[d_band.astype(bool)]

                if len(t_band) >= 2:  # Need 2 points at least for linreg
                    # Identify peak mjd and transpose series to mjd 0 at peak
                    peak_loc = np.argmax(m_band)
                    t_band -= t_band[peak_loc]

                    # Normalize series and remove unitary bias
                    m_band /= m_band[peak_loc]
                    e_band /= m_band[peak_loc]
                    m_band -= 1
                else:
                    # If band non-existent fit is impossible ie. nan
                    linreg_bands[i, j*num_feats:j*num_feats+num_feats] = np.nan

                for k, ft in enumerate(linreg_feats):

                        if ft == 'back':
                            mask = t_band <= 0
                        elif ft == 'front':
                            mask = t_band >= 0
                        else:
                            raise ValueError('Unknown linreg feature type')

                        # Incorporate error
                        # t_band /= e_band
                        # m_band /= e_band

                        t_band_partial = t_band[mask]
                        m_band_partial = m_band[mask]

                        if len(t_band_partial) >= 2: # Need 2 points at least for linreg
                            beta_1 = 1 / np.dot(t_band_partial, t_band_partial) * np.dot(t_band_partial, m_band_partial)
                        else:
                            beta_1 = np.nan

                        linreg_bands[i, j*num_feats + k] = beta_1

        # Aggregate scores across bands
        # linreg_bands = linreg_bands[:,4:-2]
        for feat_num in range(num_feats):
            linreg_array[:,feat_num] = np.nanmean(linreg_bands[:,feat_num::num_feats], axis=1)
            #linreg_array[:,2+feat_num] = np.nanstd(linreg_bands[:,feat_num::num_feats], axis=1)

        feat_arrays.append(linreg_array)

    '''
    Aggregate all feats and wrap up
    '''

    # Stack oids to feat results
    df_array = np.hstack([np.expand_dims(oids, 1)] + feat_arrays)

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
    # Also save feature names
    feat_list = list(df.columns)[1:]
    with open(save_dir+'/'+save_name.split('.h5')[0]+'.txt', 'w') as f:
        f.writelines([f'{feat_name}\n' for feat_name in feat_list])
    with open(save_dir+'/'+save_name.split('.h5')[0]+'.pkl', 'wb') as f2:
        pickle.dump(feat_list, f2, protocol=pickle.HIGHEST_PROTOCOL)


set_str = 'training'
st = time.time()
main(
    save_dir='data/'+set_str+'_feats',
    save_name=set_str+'_set_feats_r3_cesium-feats_v5.h5',
    light_curves_dir='data/'+set_str+'_cesium_curves',
    n_batches=8,
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
