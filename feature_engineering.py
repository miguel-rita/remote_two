'''
Generate standard (custom) features from saved and already grouped lc data in cesium-ready
format
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, glob, time, gc, pickle
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
import cesium.time_series, cesium.featurize
from itertools import combinations
from scipy.optimize import minimize

def atomic_worker(args):

    # Load grouped curve info
    lcs_dir, meta_dir, compute_feats = args
    with open(lcs_dir, 'rb') as f:
        lcs = pickle.load(f)
    meta = pd.read_hdf(meta_dir)
    oids = meta['object_id'].values
    rss = meta['hostgal_photoz'].values
    distmods = meta['distmod'].values

    all_oids = np.load('data/train_oids_all.npy')

    # Feat computation controls
    feat_names = []
    feat_arrays = []

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
        m_array[:,:] = np.nan

        # Compute 'm' (flux) feats
        for i,(flux_curves, err_curves, det_curves, rs) in enumerate(zip(lcs[1], lcs[2], lcs[3], rss)):
            for j,f in enumerate(feats_to_compute):
                for k,(flux_curve, err_curve, det_curve) in enumerate(zip(flux_curves, err_curves, det_curves)):

                    num_det = np.sum(det_curve)
                    dmask = det_curve.astype(bool)
                    res = np.nan

                    if rs>0:
                        flux_curve *= (rs * 1000)**2

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

        # Compute flux ratios
        # max_fluxes = m_array[:,6:12]
        # for i, (band_a, band_b) in enumerate(combinations(range(6), 2)):
        #     feat_names.append(f'maxflux_ratio_bands_{band_a:d}_{band_b:d}')
        #     temp_ratio = max_fluxes[:,[band_a]] / max_fluxes[:, [band_b]]
        #     feat_arrays.append(temp_ratio)
        #
        # mean_fluxes = m_array[:, :6]
        # for i, (band_a, band_b) in enumerate(combinations(range(6), 2)):
        #     feat_names.append(f'meanflux_ratio_bands_{band_a:d}_{band_b:d}')
        #     temp_ratio = mean_fluxes[:, [band_a]] / mean_fluxes[:, [band_b]]
        #     feat_arrays.append(temp_ratio)

    '''
    allm-feats (flux)
    '''
    if compute_feats['allm-feats']:

        # Define feats to compute
        feats_to_compute = [np.average, np.std, skew, kurtosis]
        func_names = ['all_flux_' + n for n in ['mean', 'std', 'skew', 'kurt']]
        for fn in func_names:
            feat_names.append(fn)

        # Allocate numpy placeholder for computed feats
        allm_array = np.zeros(
            shape=(oids.size, len(feat_names))
        )
        allm_array[:, :] = np.nan

        # fuse all curve data into one
        all_fused_curves = []
        for i, (ts, ms, es, ds) in enumerate(zip(lcs[0], lcs[1], lcs[2], lcs[3])):
            fused_curve_list = []
            for j, (t,m,e,d) in enumerate(zip(ts,ms,es,ds)):
                fused_curve_list.append(np.vstack([t,m,e,d]))
            fused_curve = np.hstack(fused_curve_list).T

            # Sort by mjd and append to list
            fused_curve = fused_curve[np.argsort(fused_curve[:,0])]
            all_fused_curves.append(fused_curve)

        for i, fcurve in enumerate(all_fused_curves):

            t,m,e,d = fcurve[:,0], fcurve[:,1], fcurve[:,2], fcurve[:,3]
            dmask = d.astype(bool)
            num_det = np.sum(d)

            for j, f in enumerate(feats_to_compute):
                if f == np.average:
                    res = f(m, weights=1 / e)
                elif f == np.std:
                    if num_det >= 2:
                        res = f(m[dmask])
                else:
                    res = f(m)

                allm_array[i, j] = res

        feat_arrays.append(allm_array)

    '''
    m-feats (flux)
    '''
    if compute_feats['absmag-feats']:

        # Define feats to compute
        feats_to_compute = [np.min]
        num_bands = 6
        func_names = ['abs_magnitude_'+n for n in ['max']]
        for fn in func_names:
            feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

        # Allocate numpy placeholder for computed feats
        absmag_array = np.zeros(
            shape=(oids.size, len(feat_names))
        )

        # Compute absolute magnitude feats
        for i,(flux_curves, err_curves, detec_curves, distmod_) in enumerate(zip(lcs[1], lcs[2], lcs[3], distmods)):
            for j,f in enumerate(feats_to_compute):
                for k,(flux_curve, err_curve, detec_curve) in enumerate(zip(flux_curves, err_curves, detec_curves)):

                    # Filter out negative fluxes since where looking for positive maxes
                    flux_curve = flux_curve[flux_curve > 10]

                    if distmod_ == np.nan or flux_curve.size == 0:
                        res = np.nan
                    else:
                        flux_curve = -2.5 * np.log10(flux_curve) - distmod_
                        res = f(flux_curve)

                    absmag_array[i, j * num_bands + k] = res

        feat_arrays.append(absmag_array)

        feat_names.append('abs_max_crossband_mag_agg')
        feat_arrays.append(np.nanmax(absmag_array[:,2:5], axis=1, keepdims=True))

        # for i, (band_a, band_b) in enumerate(combinations([2, 3, 4, 5], 2)):
        #     feat_names.append(f'absmagmax_ratio_bands_{band_a:d}_{band_b:d}')
        for i, (band_a, band_b) in enumerate(combinations([3, 4, 5], 2)):
            feat_names.append(f'absmagmax_diff_bands_{band_a:d}_{band_b:d}')
            temp_ratio = absmag_array[:, [band_a]] - absmag_array[:, [band_b]]
            feat_arrays.append(temp_ratio)

        # Compute cross abs_mag relations
        # cross_band_names = ['cross_band_absmag_max_contrib']
        # for fn in cross_band_names:
        #     feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

        # # For bands 2-4
        # cross_band_names = ['red_cross_band_flux_mean_contrib', 'red_cross_band_flux_max_contrib']
        # for fn in cross_band_names:
        #     feat_names.extend([f'{fn}_{i:d}' for i in [2,3,4]])

        # absmag_cross_array = np.hstack([
        #     absmag_array[:,:6] / np.sum(absmag_array[:,:6], axis=1)[:,None],
        #     # absmag_array[:,6:12] / np.sum(absmag_array[:,6:12], axis=1)[:,None],
        # ])
        # feat_arrays.append(absmag_cross_array)

    '''
    t-related feats (mjd, detected)
    '''
    if compute_feats['t-feats']:

        num_feats = 2

        # Allocate numpy placeholder for computed feats
        t_array = np.zeros(
            shape=(oids.size, num_feats)
        )

        # Define feats to compute
        def detected_feats(ts, ds):
            '''
            Compute several detected-related feats on aggregated channels:
                1) Max 1's amplitude
                2) Num. 1 blocks (Deprecated)
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
                #n_gps,
                mean_gp,
            ])

        feats_to_compute = [detected_feats]
        feat_names.extend([
            'det_amplitude',
            #'det_n_1_blocks',
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

        # for fn in func_names:
        #     feat_names.extend([f'{fn}_{i:d}' for i in range(num_bands)])

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

        # feat_arrays.extend([d_array])

        # Compute detection contribution
        feat_names.extend([f'cross_detected_contrib_{band_a:d}' for band_a in range(6)])
        cross_detect_array =  d_array / np.sum(d_array, axis=1)[:,None],
        feat_arrays.append(cross_detect_array[0])

    '''
    Peak feats
    '''
    if compute_feats['peak-feats']:

        # peak_feats = ['max_delta_peak']
        # feat_names.extend([f'{fn}' for fn in peak_feats])

        # Temp array for later collapse
        peak_array_bands_max = np.zeros(
            shape=(oids.size, 6)  # six bands per feat
        )
        peak_array_bands_mean = np.zeros(
            shape=(oids.size, 6)  # six bands per feat
        )

        # Compute simple b1 from linreg from normalized peak
        for i, (t_curves, m_curves, e_curves, d_curves) in enumerate(zip(lcs[0], lcs[1], lcs[2], lcs[3])):
            for j, (t_band, m_band, e_band, d_band) in enumerate(zip(t_curves, m_curves, e_curves, d_curves)):

                # Filtered only
                m_band = m_band[d_band.astype(bool)]
                t_band = t_band[d_band.astype(bool)]
                e_band = e_band[d_band.astype(bool)]

                if len(t_band) > 0:  # Need at least 1 point to compute peak
                    peak_array_bands_max[i, j] = t_band[np.argmax(m_band)]
                    peak_array_bands_mean[i, j] = np.mean(t_band)
                else:
                    peak_array_bands_max[i, j] = np.nan
                    peak_array_bands_mean[i, j] = np.nan

        #peak_array_bands = peak_array_bands[:,2:5]
        #peak_array = np.nanmax(peak_array_bands, axis=1, keepdims=True) - np.nanmin(peak_array_bands, axis=1, keepdims=True)

        peak_array_max = np.zeros(shape=(oids.size, 15))
        peak_array_mean = np.zeros(shape=(oids.size, 15))

        # Compute cross band max peak mjd deltas
        for i, (band_a, band_b) in enumerate(combinations(range(6), 2)):
            peak_array_max[:, i] = peak_array_bands_max[:,band_a] - peak_array_bands_max[:,band_b]
            peak_array_mean[:, i] = peak_array_bands_mean[:,band_a] - peak_array_bands_mean[:,band_b]

            feat_names.append(f'mjd_peak_dist_max_{band_a:d}_{band_b:d}')
            feat_names.append(f'mjd_peak_dist_mean_{band_a:d}_{band_b:d}')

        feat_arrays.extend([peak_array_max, peak_array_mean])

    '''
    SN feats
    '''
    if compute_feats['sn-feats']:

        # Load fit data
        sn_fit = np.load('data/train_sn_fits.npy')

        # Params
        ERROR_THRESH = 1500
        BANDS = [0,1,2,3,4,5]

        # Temp array for later collapse
        sn_feat_arr = np.zeros(
            shape=(oids.size, 6)  # six bands per feat
        )
        sn_feat_arr[:,:] = np.nan

        for i, (t_curves, m_curves, e_curves, d_curves, rs_, oid_, distmod_) in enumerate(zip(lcs[0], lcs[1], lcs[2], lcs[3], rss, oids, distmods)):
            for j, (t_band, m_band, e_band, d_band) in enumerate(zip(t_curves, m_curves, e_curves, d_curves)):

                dmask = d_band.astype(bool)

                # Filtered only
                m_band = m_band[dmask]
                t_band = t_band[dmask]

                if len(t_band) >= 2 and np.logical_not(
                        np.any(m_band < 0)) and rs_ > 0:

                    # Get fit info
                    oid_pos = np.argwhere(all_oids == oid_)[0,0]
                    a0, t0, trise, tfall, msqe = tuple(sn_fit[oid_pos, j, :])

                    if msqe <= ERROR_THRESH:
                        # print('t0:',t0,'targmax:',t_band[np.argmax(m_band)],' oid:',oid_)
                        sn_feat_arr[i,j] = -2.5*np.log10(a0) - distmod_

        feat_names.extend([f'log_a0_band{b:d}' for b in BANDS])
        feat_arrays.extend([sn_feat_arr[:,BANDS]])

    '''
    Linreg feats
    '''
    if compute_feats['linreg-feats']:

        BANDS = [2,3,4]
        linreg_ranges = [(0,110)]
        num_feats = len(BANDS) * len(linreg_ranges)

        # Band info array for later collapse
        linreg_bands = np.zeros(
            shape=(oids.size, num_feats)  # six bands per feat
        )
        linreg_bands[:,:] = np.nan

        linreg_bands_raw = np.zeros(
            shape=(oids.size, num_feats)  # six bands per feat
        )
        linreg_bands_raw[:,:] = np.nan

        # Compute simple b1 from linreg from normalized peak
        for i, (t_curves, m_curves, e_curves, d_curves, rs_, distmod_,) in enumerate(zip(lcs[0], lcs[1], lcs[2], lcs[3], rss, distmods,)):
            for j, (t_band, m_band, e_band, d_band) in enumerate(zip([t_curves[b] for b in BANDS], [m_curves[b] for b in BANDS], [e_curves[b] for b in BANDS], [d_curves[b] for b in BANDS])):

                # Filtered only
                m_band = m_band[d_band.astype(bool)]
                t_band = t_band[d_band.astype(bool)]

                if len(t_band) >= 2 and np.logical_not(np.any(m_band < 0)) and rs_>0:  # Need 2 points at least for linreg
                    # Identify peak mjd and transpose series to mjd 0 at peak
                    peak_loc = np.argmax(m_band)
                    t_band -= t_band[peak_loc]
                else:
                    # If band non-existent fit is impossible
                    continue


                for k, rg in enumerate(linreg_ranges):

                        mask = (t_band >= rg[0]) & (t_band <= rg[1])

                        # Incorporate error
                        # t_band /= e_band
                        # m_band /= e_band

                        t_band_partial = t_band[mask]
                        m_band_partial = m_band[mask]

                        if len(t_band_partial) >= 2: # Need 2 points at least for linreg

                            X = np.ones((t_band_partial.shape[0], 2))
                            X[:,1] = t_band_partial

                            m_band_partial_raw_backup = np.copy(m_band_partial)
                            m_band_partial = -2.5 * np.log10(m_band_partial) - distmod_

                            y = m_band_partial
                            betas = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
                            beta_1 = betas[1]

                            # Raw coefficients
                            y_raw = m_band_partial_raw_backup
                            betas_raw = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y_raw)
                            beta_1_raw = betas_raw[1]

                        else:
                            beta_1 = np.nan
                            beta_1_raw = np.nan

                        linreg_bands[i, j*len(linreg_ranges) + k] = beta_1
                        linreg_bands_raw[i, j*len(linreg_ranges) + k] = beta_1_raw

        # Abs mag slopes
        for bn in BANDS:
            for rg in linreg_ranges:
                feat_names.append(f'linreg_b1_{rg[0]:d}_{rg[1]:d}_band{bn:d}')

        feat_arrays.append(linreg_bands)

        # # Cross ratios
        # beta_ratios = np.zeros((oids.size, 4)) #4 bands
        # beta_ratios[:,:] = np.nan
        #
        # # beta_ratios = linreg_bands_raw[:,1::3] / linreg_bands_raw[:,2::3]
        # # for b in [2,3,4,5]:
        # #     feat_names.append(f'raw_b1_ratio_band{b:d}')
        #
        # feat_names.append('mean_raw_b1_ratio')
        #
        # feat_arrays.append(np.nanmean(beta_ratios,axis=1,keepdims=True))
        # Compute cross ratios
        # linreg_ratios = np.zeros(
        #     shape=(oids.size, len(BANDS))
        # )
        #
        # linreg_ratios[:,0] = linreg_bands[:,2] - linreg_bands[:,0]
        # linreg_ratios[:,1] = linreg_bands[:,2+3] - linreg_bands[:,0+3]
        # linreg_ratios[:,2] = linreg_bands[:,2+3*2] - linreg_bands[:,0+3*2]
        #
        # feat_names.extend([f'half_ratio_band{bn:d}' for bn in BANDS])
        #
        # feat_arrays.append(linreg_ratios)

        # # Compute cross relations
        # linreg_cross_array = np.zeros((oids.size, 6*2))
        # # Add names
        # for fn in ['linregback', 'linregfront']:
        #     feat_names.extend([f'{fn}_{j}' for j in range(6)])
        # # Compute
        # for feat_num in range(num_feats):
        #     linreg_cross_array[:,feat_num*6:(feat_num+1)*6] =\
        #         linreg_bands[:,feat_num::num_feats] /\
        #         np.nansum(np.abs(linreg_bands[:, feat_num::num_feats]), axis=1, keepdims=True)
        # # Append cross relations
        # feat_arrays.append(linreg_cross_array)

    '''
    Expreg feats
    '''
    if compute_feats['expreg-feats']:

        def exp_model(t, decay):
            return np.e ** (-t/decay)

        def msqe(x0, *args):
            ts, ys = args
            return np.mean((exp_model(ts, x0) - ys) ** 2)

        BANDS = [2, 3, 4]
        expreg_ranges = [(0, 110)]
        num_feats = len(BANDS) * len(expreg_ranges)

        # Band info array for later collapse
        expreg_bands = np.zeros(
            shape=(oids.size, num_feats)  # six bands per feat
        )
        expreg_bands[:, :] = np.nan
        expreg_msqes = np.copy(expreg_bands)

        # Compute expdecay rates from normalize curves
        for i, (t_curves, m_curves, e_curves, d_curves, rs_, distmod_,) in tqdm.tqdm(enumerate(
                zip(lcs[0], lcs[1], lcs[2], lcs[3], rss, distmods, )), total=len(lcs[0])):
            for j, (t_band, m_band, e_band, d_band) in enumerate(
                    zip([t_curves[b] for b in BANDS], [m_curves[b] for b in BANDS], [e_curves[b] for b in BANDS],
                        [d_curves[b] for b in BANDS])):

                # Filtered only
                m_band = m_band[d_band.astype(bool)]
                t_band = t_band[d_band.astype(bool)]

                if len(t_band) >= 2 and np.logical_not(
                        np.any(m_band < 0)) and rs_ > 0:  # Need 2 points at least for linreg
                    # Identify peak mjd and transpose series to mjd 0 at peak, normalize
                    peak_loc = np.argmax(m_band)
                    t_band -= t_band[peak_loc]
                    m_band /= m_band[peak_loc]
                else:
                    # If band non-existent fit is impossible
                    continue

                for k, rg in enumerate(expreg_ranges):

                    mask = (t_band >= rg[0]) & (t_band <= rg[1])

                    t_band_partial = t_band[mask]
                    m_band_partial = m_band[mask]

                    if len(t_band_partial) >= 2:

                        x = t_band_partial
                        y = m_band_partial

                        # Fit exp model
                        x0 = np.array([-t_band_partial[-1] / np.log(m_band_partial[-1])]) # -150 days / ln(0.4 of max)
                        bnds = [(12, 800)]
                        args = (t_band_partial, m_band_partial)
                        res = minimize(msqe, x0, args, bounds=bnds)

                        decay = res.x[0]
                        msqe_ = msqe(decay, *args)

                        expreg_bands[i, j * len(expreg_ranges) + k] = decay
                        expreg_msqes[i, j * len(expreg_ranges) + k] = msqe_


        # Decay rates per band
        # for bn in BANDS:
        #     for rg in expreg_ranges:
        #         feat_names.append(f'expreg_decay_{rg[0]:d}_{rg[1]:d}_band{bn:d}')

        feat_names.extend(['mean_decay_rate', 'min_decay_rate'])
        arr1 = np.nanmean(expreg_bands,axis=1,keepdims=True)
        arr2 = np.nanmin(expreg_bands,axis=1,keepdims=True)

        feat_arrays.extend([arr1, arr2])

        # feat_arrays.append(expreg_bands)

    '''
    Adim linreg feats
    '''
    if compute_feats['adim-linreg-feats']:

        ctypes = ['peak', 'tail']
        stat_names = ['maxclimb']
        BANDS = [1,2,3,4,5]
        linreg_ranges = [(40, 190)]
        num_feats = len(ctypes) * len(stat_names) * len(BANDS) * len(linreg_ranges)

        for tp in ctypes:
            for b in BANDS:
                for sn in stat_names:
                    for r in linreg_ranges:
                        if tp == 'peak':
                            # feat_names.append(f'alr_{tp}_band{b:d}_{sn}_range{r[0]:d}_{r[1]:d}')
                            continue

        # Full feat array
        alr_arr = np.zeros(
            shape=(oids.size, num_feats)
        )
        alr_arr[:,:] = np.nan # Since we'll have tail OR peak feats, we must forcibly have nans

        # Compute adimensional feats
        for i, (t_curves, m_curves, e_curves, d_curves, rs_, distmod_,) in enumerate(
                zip(lcs[0], lcs[1], lcs[2], lcs[3], rss, distmods, )):
            for j, (t_band, m_band, e_band, d_band) in enumerate(
                    zip([t_curves[b] for b in BANDS], [m_curves[b] for b in BANDS], [e_curves[b] for b in BANDS],
                        [d_curves[b] for b in BANDS])):

                dmask = d_band.astype(bool)

                # Filtered only
                orig_m_band = m_band
                orig_t_band = t_band
                m_band = m_band[dmask]
                t_band = t_band[dmask]

                if len(t_band) >= 2 and np.logical_not(
                        np.any(m_band < 0)) and rs_ > 0:  # Need 2 points at least for linreg

                    # Determine if peak or tail - nan if other
                    if np.sum(d_band) == 1:  # Single pt
                        ctp = 3
                    elif not np.any(m_band > m_band[-1]):
                        ctp = 0  # Head
                    elif not np.any(m_band > m_band[0]):
                        ctp = 2  # Tail
                    else:
                        ctp = 1  # Peak

                    if ctp in [1,2]: # If peak or tail
                        # Identify peak mjd, transpose series to mjd 0 at peak, invert and normalize (fountain)
                        peak_loc = np.argmax(m_band)
                        t_band -= t_band[peak_loc]
                        m_band /= m_band[peak_loc]
                        # m_band = -m_band + 1
                    else:
                        continue
                else:
                    continue

                for h, stat_name in enumerate(stat_names):
                    for k, rg in enumerate(linreg_ranges):

                        # Force tails disguised as peaks - we know they're tails due to magnitude near noise level
                        if ctp == 1:
                            ORDER_OF_MAG_CONST = 3
                            if np.sum(d_band) < d_band.size:  # If we have non detected points too (0s besides 1s) - percentile to handle noise outliers
                                if np.max(orig_m_band[dmask]) < ORDER_OF_MAG_CONST * np.percentile(orig_m_band[np.logical_not(dmask)], 85):
                                    ctp = 2

                        # If we're on a tail we'll use the right part of alr_arr
                        #arr_offset = 0 if ctp==1 else num_feats/2
                        # GROUP EVERYTHING
                        arr_offset = 0

                        # Apply range mask
                        mask = (t_band >= rg[0]) & (t_band <= rg[1])

                        t_band_partial = t_band[mask]
                        m_band_partial = m_band[mask]

                        if stat_name == 'beta-amp':

                            if len(t_band_partial) >= 2:  # Need 2 points at least for linreg

                                # Get beta estim.
                                X = t_band_partial[:,None]
                                y = m_band_partial
                                beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

                                # Compute fit msqe
                                #msqe = np.mean((X * beta - y)**2)

                                # Assign to feat arr
                                ix = int(arr_offset + j*num_feats/len(ctypes)/len(BANDS) +\
                                              h*num_feats/len(ctypes)/len(BANDS)/len(stat_names) + k)
                                alr_arr[i, ix] = beta

                        elif stat_name == 'maxmin_declinerate':

                            if len(t_band_partial) >= 2:  # Need 2 points at least for linreg

                                ix = int(arr_offset + j * num_feats / len(ctypes) / len(BANDS) + \
                                         h * num_feats / len(ctypes) / len(BANDS) / len(stat_names) + k)

                                alr_arr[i, ix] = (m_band_partial[-1] - m_band_partial[0]) / (t_band_partial[-1] - t_band_partial[0])

                        elif stat_name == 'density_local_maxima':

                            if len(t_band_partial) >= 3:

                                peak_ixs, _ = find_peaks(m_band_partial)

                                ix = int(arr_offset + j * num_feats / len(ctypes) / len(BANDS) + \
                                         h * num_feats / len(ctypes) / len(BANDS) / len(stat_names) + k)

                                alr_arr[i, ix] = len(peak_ixs) / len(t_band_partial)
                        elif stat_name == 'maxclimb':

                            if len(t_band_partial) >= 8:

                                max_strk = 0
                                curr_strk = 0

                                for pti, pt in enumerate(m_band_partial):
                                    if pti == 0:
                                        continue
                                    delta = pt - m_band_partial[pti-1]
                                    if delta >= 0:
                                        curr_strk += delta
                                    else:
                                        max_strk = np.max([curr_strk, max_strk])
                                        curr_strk = 0
                                if curr_strk != 0:
                                    max_strk = np.max([curr_strk, max_strk])

                                ix = int(arr_offset + j * num_feats / len(ctypes) / len(BANDS) + \
                                         h * num_feats / len(ctypes) / len(BANDS) / len(stat_names) + k)

                                alr_arr[i, ix] = max_strk
                        else:
                            print('TODO')

        # Temp
        # feat_names.extend(['alr_beta-amp'])
        # feat_arrays.append(np.nanmax(alr_arr, axis=1, keepdims=True) - np.nanmin(alr_arr, axis=1, keepdims=True))

        # feat_arrays.append(alr_arr[:,:5])

        # maxclimb streak

        arr = alr_arr[:,:5]
        maxarr = np.nanmax(arr, axis=1, keepdims=True)
        feat_arrays.append(maxarr)
        feat_names.append('maxclimb_streak')

    '''
    Maxima feats
    '''
    if compute_feats['maxima-feats']:

        # def perc_ratio(df, half_perc):
        #     num = np.percentile(df, 50+half_perc) - np.percentile(df, 50-half_perc)
        #     return num / (np.percentile(df, 95) - np.percentile(df, 5))
        #
        # half_percs = [10, 18, 25, 40]

        BANDS = [4,5]
        # for band in BANDS:
        #     for fn in ['maxptp']:
        #         feat_names.append(f'time_maxima_band{band:d}_{fn}')

        # Full feat array
        maxima_arr = np.zeros(
            shape=(oids.size, len(BANDS))
        )
        maxima_arr[:, :] = np.nan

        # Compute perc ratio feats
        for i, (t_curves, m_curves, e_curves, d_curves, rs_, distmod_,) in enumerate(
                zip(lcs[0], lcs[1], lcs[2], lcs[3], rss, distmods, )):
            for j, (t_band, m_band, e_band, d_band) in enumerate(
                    zip([t_curves[b] for b in BANDS], [m_curves[b] for b in BANDS], [e_curves[b] for b in BANDS],
                        [d_curves[b] for b in BANDS])):

                dmask = d_band.astype(bool)

                # Filtered only
                m_band = m_band[dmask]
                t_band = t_band[dmask]

                peak_ixs, _ = find_peaks(m_band)

                # Check if first and last pts are peaks
                if np.sum(d_band) >= 2:
                    if m_band[0] > m_band[1]:
                        peak_ixs = np.hstack([[0], peak_ixs])
                    if m_band[-1] > m_band[-2]:
                        peak_ixs = np.hstack([peak_ixs, [-1]])

                peaks = m_band[peak_ixs]
                tpeaks = t_band[peak_ixs]

                if peaks.size >= 2:
                    maxima_arr[i, j] = np.max(tpeaks[1:]-tpeaks[:-1]) / np.abs(np.max(peaks[1:]-peaks[:-1]))

        feat_names.append('max_deltat_peaks_allb')
        feat_arrays.append(np.nanmax(maxima_arr, axis=1, keepdims=True))

    '''
    Shoulder feats
    '''
    if compute_feats['shoulder-feats']:

        BANDS = [4,5]
        num_feats = len(BANDS)
        feat_names.extend(['shoulderness_band5'])

        # Band info array for later collapse
        shoulder_arr = np.zeros(
            shape=(oids.size, len(BANDS))  # 3 bands for later average
        )



        # Compute simple b1 from linreg from normalized peak
        for i, (t_curves, m_curves, e_curves, d_curves, rs_, distmod_) in enumerate(zip(lcs[0], lcs[1], lcs[2], lcs[3], rss, distmods)):
            for j, (t_band, m_band, e_band, d_band) in enumerate(zip([t_curves[b] for b in BANDS], [m_curves[b] for b in BANDS], [e_curves[b] for b in BANDS], [d_curves[b] for b in BANDS])):

                # Filtered only
                m_band = m_band[d_band.astype(bool)]
                t_band = t_band[d_band.astype(bool)]
                e_band = e_band[d_band.astype(bool)]

                if len(t_band) >= 3 and np.logical_not(np.any(m_band < 0)) and rs_>0:  # Need 3 pts for 2nd derivative
                    # Identify peak mjd and transpose series to mjd 0 at peak
                    peak_loc = np.argmax(m_band)
                    t_band -= t_band[peak_loc]
                    m_band /= m_band[peak_loc]
                else:
                    # If band non-existent fit is impossible ie. nan
                    shoulder_arr[i, j*num_feats:j*num_feats+num_feats] = np.nan
                    continue

                # Compute deltas to the right of maximum
                m_band = m_band[peak_loc:]
                t_band = t_band[peak_loc:]
                deltas = (m_band[1:] - m_band[:-1]) / (t_band[1:] - t_band[:-1])

                # Compute sum delta2
                deltas2 = deltas[1:] - deltas[:-1]
                shoulder_arr[i, j * num_feats:j * num_feats + num_feats] = np.sum(deltas2)

        all_bands_shoulder_array = np.nanmean(shoulder_arr, axis=1, keepdims=True)
        feat_arrays.append(all_bands_shoulder_array)

        # Compute cross ratios
        # linreg_ratios = np.zeros(
        #     shape=(oids.size, len(BANDS))
        # )
        #
        # linreg_ratios[:,0] = linreg_bands[:,2] - linreg_bands[:,0]
        # linreg_ratios[:,1] = linreg_bands[:,2+3] - linreg_bands[:,0+3]
        # linreg_ratios[:,2] = linreg_bands[:,2+3*2] - linreg_bands[:,0+3*2]
        #
        # feat_names.extend([f'half_ratio_band{bn:d}' for bn in BANDS])
        #
        # feat_arrays.append(linreg_ratios)

        # # Compute cross relations
        # linreg_cross_array = np.zeros((oids.size, 6*2))
        # # Add names
        # for fn in ['linregback', 'linregfront']:
        #     feat_names.extend([f'{fn}_{j}' for j in range(6)])
        # # Compute
        # for feat_num in range(num_feats):
        #     linreg_cross_array[:,feat_num*6:(feat_num+1)*6] =\
        #         linreg_bands[:,feat_num::num_feats] /\
        #         np.nansum(np.abs(linreg_bands[:, feat_num::num_feats]), axis=1, keepdims=True)
        # # Append cross relations
        # feat_arrays.append(linreg_cross_array)

    '''
    Flag feats
    '''
    if compute_feats['flag-feats']:

        feat_names.extend(['is_tail'])

        BANDS = [2,3,4,5]

        # Band info array for later collapse
        flag_arr_all_bands = np.zeros(
            shape=(oids.size, len(BANDS))  # 3 bands for later average
        )
        flag_arr_all_bands[:,:] = np.nan

        for i, (t_curves, m_curves, e_curves, d_curves, rs_, distmod_,) in enumerate(
                zip(lcs[0], lcs[1], lcs[2], lcs[3], rss, distmods, )):
            for j, (t_band, m_band, e_band, d_band) in enumerate(
                    zip([t_curves[b] for b in BANDS], [m_curves[b] for b in BANDS], [e_curves[b] for b in BANDS],
                        [d_curves[b] for b in BANDS])):

                dmask = d_band.astype(bool)

                # Filtered only
                orig_m_band = m_band
                orig_t_band = t_band
                m_band = m_band[dmask]
                t_band = t_band[dmask]

                if len(t_band) >= 2 and np.logical_not(
                        np.any(m_band < 0)) and rs_ > 0:  # Need 2 points at least for linreg

                    # Determine if peak or tail - nan if other
                    if np.sum(d_band) == 1:  # Single pt
                        ctp = 3
                    elif not np.any(m_band > m_band[-1]):
                        ctp = 0  # Head
                    elif not np.any(m_band > m_band[0]):
                        ctp = 2  # Tail
                    else:
                        ctp = 1  # Peak

                    # Force tails disguised as peaks - we know they're tails due to magnitude near noise level
                    if ctp == 1:
                        ORDER_OF_MAG_CONST = 5
                        if np.sum(
                                d_band) < d_band.size:  # If we have non detected points too (0s besides 1s) - percentile to handle noise outliers
                            if np.max(orig_m_band[dmask]) < ORDER_OF_MAG_CONST * np.percentile(
                                    orig_m_band[np.logical_not(dmask)], 85):
                                ctp = 2

                    flag_arr_all_bands[i,j] = ctp

        # Collapse flag by majority vote - if tie pick lowest ctp from among tie
        majority_flag = np.zeros((oids.size, 1))
        for fi, fline in enumerate(flag_arr_all_bands):
            u,c = np.unique(fline, return_counts=True)
            majority_flag[fi,0] = u[np.argmax(c)]
        flag_arr = majority_flag == 2
        flag_arr = flag_arr.astype(int)
        feat_arrays.append(flag_arr)

        # Compute cross ratios
        # linreg_ratios = np.zeros(
        #     shape=(oids.size, len(BANDS))
        # )
        #
        # linreg_ratios[:,0] = linreg_bands[:,2] - linreg_bands[:,0]
        # linreg_ratios[:,1] = linreg_bands[:,2+3] - linreg_bands[:,0+3]
        # linreg_ratios[:,2] = linreg_bands[:,2+3*2] - linreg_bands[:,0+3*2]
        #
        # feat_names.extend([f'half_ratio_band{bn:d}' for bn in BANDS])
        #
        # feat_arrays.append(linreg_ratios)

        # # Compute cross relations
        # linreg_cross_array = np.zeros((oids.size, 6*2))
        # # Add names
        # for fn in ['linregback', 'linregfront']:
        #     feat_names.extend([f'{fn}_{j}' for j in range(6)])
        # # Compute
        # for feat_num in range(num_feats):
        #     linreg_cross_array[:,feat_num*6:(feat_num+1)*6] =\
        #         linreg_bands[:,feat_num::num_feats] /\
        #         np.nansum(np.abs(linreg_bands[:, feat_num::num_feats]), axis=1, keepdims=True)
        # # Append cross relations
        # feat_arrays.append(linreg_cross_array)

    '''
    Spike feats
    '''
    if compute_feats['spike-feats']:

        spike_feats = ['all']
        agg_feats = ['mean']  # , 'std']
        num_feats = len(spike_feats)  # Back and front linreg fits

        # Allocate numpy placeholder for computed feats
        spike_array = np.zeros(
            shape=(oids.size, num_feats * len(agg_feats))
        )

        local_feat_names = [f'spike_{cn_}' for cn_ in spike_feats]
        for j in agg_feats:
            feat_names.extend([f'{fn}_{j}' for fn in local_feat_names])

        # Band info array for later collapse
        spike_bands = np.zeros(
            shape=(oids.size, num_feats * 6)  # six bands per feat
        )

        # Compute spike feat from normalized peak
        for i, (t_curves, m_curves, e_curves, d_curves) in enumerate(zip(lcs[0], lcs[1], lcs[2], lcs[3])):
            for j, (t_band, m_band, e_band, d_band) in enumerate(zip(t_curves, m_curves, e_curves, d_curves)):

                # Filtered only
                m_band = m_band[d_band.astype(bool)]
                t_band = t_band[d_band.astype(bool)]
                e_band = e_band[d_band.astype(bool)]

                if len(t_band) >= 2:  # Need 2 points at least for spyke
                    # Identify peak mjd and transpose series to mjd 0 at peak
                    peak_loc = np.argmax(m_band)
                    t_band -= t_band[peak_loc]

                    # Normalize series and remove unitary bias
                    m_band /= m_band[peak_loc]
                    e_band /= m_band[peak_loc]
                    m_band -= 1
                else:
                    # If band non-existent fit is impossible ie. nan
                    spike_bands[i, j * num_feats:j * num_feats + num_feats] = np.nan

                for k, ft in enumerate(spike_feats):

                    if ft == 'back':
                        mask = (t_band <= 0)
                    elif ft == 'front':
                        mask = (t_band >= 0)
                    else:
                        mask = t_band <= 1e20

                    # Incorporate error
                    # t_band /= e_band
                    # m_band /= e_band

                    t_band_partial = t_band[mask]
                    m_band_partial = m_band[mask]

                    if len(t_band_partial) >= 2:  # Need 2 points at least for spike

                        deltas = m_band_partial[1:]-m_band_partial[:-1]
                        spike = np.sum(np.abs(deltas)) / np.max([0.05, np.abs(np.sum(deltas))])

                    else:
                        spike = np.nan

                    spike_bands[i, j * num_feats + k] = spike

        # Aggregate scores across bands
        # linreg_bands = linreg_bands[:,4:-2]
        for feat_num in range(num_feats):
            spike_array[:, feat_num] = np.nanmean(spike_bands[:, feat_num::num_feats], axis=1)
            # linreg_array[:,2+feat_num] = np.nanstd(linreg_bands[:,feat_num::num_feats], axis=1)

        feat_arrays.append(spike_array)

        # # Compute cross relations
        # linreg_cross_array = np.zeros((oids.size, 6*2))
        # # Add names
        # for fn in ['linregback', 'linregfront']:
        #     feat_names.extend([f'{fn}_{j}' for j in range(6)])
        # # Compute
        # for feat_num in range(num_feats):
        #     linreg_cross_array[:,feat_num*6:(feat_num+1)*6] =\
        #         linreg_bands[:,feat_num::num_feats] /\
        #         np.nansum(np.abs(linreg_bands[:, feat_num::num_feats]), axis=1, keepdims=True)
        # # Append cross relations
        # feat_arrays.append(linreg_cross_array)

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
    Aggregate all feats and wrap up
    '''

    # Build final pandas dataframe
    df = pd.DataFrame(
        data=np.hstack(feat_arrays),
        columns=feat_names,
    )
    # Concat oids to feat results
    df.insert(0, 'object_id', oids)

    del lcs
    gc.collect()

    return df


def gen_feats(save_dir, save_name, light_curves_dir, n_batches, compute_feats):
    '''
    Generate custom features data frame from stored grouped light curve data

    :param save_dir (str) Dir to save calculated feats
    :param save_name (str) Feat set name
    :param light_curves_dir (str) 
    :param n_batches (int) Process using multiprocess on one batch of saved lc data at a time
    :param compute_feats (dict) Dict of bools marking the feats to generate
    :return:
    '''

    np.warnings.filterwarnings('ignore')

    # Get paths to lcs and respective oids
    atomic_args = []
    for lcs_path, meta_path in zip(sorted(glob.glob(light_curves_dir + '/*.pkl')), sorted(glob.glob(light_curves_dir + '/*.h5'))):
        atomic_args.append((lcs_path, meta_path, compute_feats))

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
    types_dict['object_id'] = np.uint64
    df = df.astype(types_dict)

    df.reset_index(drop=True).to_hdf(save_dir+'/'+save_name, key='w')
    # Also save feature names
    feat_list = list(df.columns)[1:]
    with open(save_dir+'/'+save_name.split('.h5')[0]+'.txt', 'w') as f:
        f.writelines([f'{feat_name}\n' for feat_name in feat_list])
    with open(save_dir+'/'+save_name.split('.h5')[0]+'.pkl', 'wb') as f2:
        pickle.dump(feat_list, f2, protocol=pickle.HIGHEST_PROTOCOL)


set_str = 'test'
st = time.time()

compute_feats_template = {
    'm-feats':          bool(0),
    'allm-feats':       bool(0),
    't-feats':          bool(0),
    'd-feats':          bool(0),
    'linreg-feats':     bool(0),
    'expreg-feats':     bool(0),
    'adim-linreg-feats':bool(0),
    'absmag-feats':     bool(0),
    'sn-feats':         bool(0),
    'maxima-feats':     bool(0),
    'shoulder-feats':   bool(0),
    'peak-feats':       bool(0),
    'spike-feats':      bool(0),
    'e-feats':          bool(0),
    'cesium-feats':     bool(0),
    'slope-feats':      bool(0),
    'curve-feats':      bool(0),
    'flag-feats':       bool(0),
}

feats_to_gen = {
    'm-feats': 'm-feats_v6',
    # 'allm-feats': 'allm-feats_v1',
    # 'absmag-feats': 'absmag-feats_v5',
    # 't-feats': 't-feats_v1',
    # 'd-feats': 'd-feats_v2',
    # 'expreg-feats': 'expreg-feats_v3'
    # 'linreg-feats': 'linreg-feats_v5',
    # 'adim-linreg-feats': 'adim-linreg-feats_v3'
    # 'sn-feats': 'sn-feats_v1'
    # 'maxima-feats': 'maxima-feats_v1'
    # 'shoulder-feats': 'shoulder-feats_v1'
    # 'flag-feats': 'flag-feats_v1'
    # 'spike-feats': 'spike-feats_v1'
}

for ft_name, file_name in feats_to_gen.items():

    cpt_fts = compute_feats_template.copy()
    cpt_fts[ft_name] = True

    gen_feats(
        save_dir='data/'+set_str+'_feats',
        save_name=set_str+f'_set_feats_r7_{file_name}.h5',
        light_curves_dir='data/'+set_str+'_cesium_curves',
        n_batches=8,
        compute_feats=cpt_fts,
    )

print(f'>   featgen_standard_v2 : Wall time : {(time.time()-st):.2f} seconds')

