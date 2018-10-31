import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, glob, time, gc, pickle
from cesium.featurize import featurize_time_series

def atomic_worker(args):

    feat_names = [
        'amplitude',
        #     'flux_percentile_ratio_mid20',
        #     'flux_percentile_ratio_mid35',
        #     'flux_percentile_ratio_mid50',
        #     'flux_percentile_ratio_mid65',
        #     'flux_percentile_ratio_mid80',
        'max_slope',
        'maximum',
        'median',
        'median_absolute_deviation',
        'minimum',
        #     'percent_amplitude',
        'percent_beyond_1_std',
        'percent_close_to_median',
        #     'percent_difference_flux_percentile',
        'period_fast',
        'qso_log_chi2_qsonu',
        'qso_log_chi2nuNULL_chi2nu',
        'skew',
        'std',
        'stetson_j',
        'stetson_k',
        'weighted_average',
        'all_times_nhist_numpeaks',
        'all_times_nhist_peak1_bin',
        'all_times_nhist_peak2_bin',
        'all_times_nhist_peak3_bin',
        'all_times_nhist_peak4_bin',
        'all_times_nhist_peak_1_to_2',
        'all_times_nhist_peak_1_to_3',
        'all_times_nhist_peak_1_to_4',
        'all_times_nhist_peak_2_to_3',
        'all_times_nhist_peak_2_to_4',
        'all_times_nhist_peak_3_to_4',
        'all_times_nhist_peak_val',
        'avg_double_to_single_step',
        'avg_err',
        'avgt',
        'cad_probs_1',
        'cad_probs_10',
        'cad_probs_20',
        'cad_probs_30',
        'cad_probs_40',
        'cad_probs_50',
        'cad_probs_100',
        'cad_probs_500',
        'cad_probs_1000',
        'cad_probs_5000',
        'cad_probs_10000',
        'cad_probs_50000',
        'cad_probs_100000',
        'cad_probs_500000',
        'cad_probs_1000000',
        'cad_probs_5000000',
        'cad_probs_10000000',
        'cads_avg',
        'cads_med',
        'cads_std',
        'mean',
        'med_double_to_single_step',
        'med_err',
        'n_epochs',
        'std_double_to_single_step',
        'std_err',
        'total_time',
        'fold2P_slope_10percentile',
        'fold2P_slope_90percentile',
        'freq1_amplitude1',
        'freq1_amplitude2',
        'freq1_amplitude3',
        'freq1_amplitude4',
        'freq1_freq',
        'freq1_lambda',
        'freq1_rel_phase2',
        'freq1_rel_phase3',
        'freq1_rel_phase4',
        'freq1_signif',
        'freq2_amplitude1',
        'freq2_amplitude2',
        'freq2_amplitude3',
        'freq2_amplitude4',
        'freq2_freq',
        'freq2_rel_phase2',
        'freq2_rel_phase3',
        'freq2_rel_phase4',
        'freq3_amplitude1',
        'freq3_amplitude2',
        'freq3_amplitude3',
        'freq3_amplitude4',
        'freq3_freq',
        'freq3_rel_phase2',
        'freq3_rel_phase3',
        'freq3_rel_phase4',
        'freq_amplitude_ratio_21',
        'freq_amplitude_ratio_31',
        'freq_frequency_ratio_21',
        'freq_frequency_ratio_31',
        'freq_model_max_delta_mags',
        'freq_model_min_delta_mags',
        'freq_model_phi1_phi2',
        'freq_n_alias',
        'freq_signif_ratio_21',
        'freq_signif_ratio_31',
        'freq_varrat',
        'freq_y_offset',
        'linear_trend',
        'medperc90_2p_p',
        'p2p_scatter_2praw',
        'p2p_scatter_over_mad',
        'p2p_scatter_pfold_over_mad',
        'p2p_ssqr_diff_over_var',
        'scatter_res_raw',
    ]

    lcs_dir, oids_dir = args

    with open(lcs_dir, 'rb') as f:
        lcs = pickle.load(f)

    # Compute feats
    #lim=2
    cesium_feats = featurize_time_series(lcs[0], lcs[1], lcs[2], features_to_use=feat_names)

    # Insert oids to identify each light curve
    cesium_feats.insert(
        loc=0,
        column='object_id',
        value=np.load(oids_dir).astype(np.uint32),
    )

    del lcs
    gc.collect()

    return cesium_feats


def main(save_dir, save_name, light_curves_dir, n_batches):
    '''
    Generate cesium features data frame from stored grouped light curve data

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

    print(f'>   featgen_cesium : Starting batch multiprocessing . . .')
    res = []
    for ixs in tqdm.tqdm(batch_split_indexes, total=len(batch_split_indexes), postfix='Batch'):
        pool = mp.Pool(processes=mp.cpu_count())
        res.extend(pool.map(atomic_worker, [atomic_args[ix] for ix in ixs]))
        pool.close()
        pool.join()

    print(f'>   featgen_cesium : Concating and saving results . . .')

    # Concat atomic computed feats, sort by oid, flatten col index, reset types, ready to test
    df = pd.concat(res, axis=0).sort_values('object_id')


    flat_index = [f'{t[0]}_{t[1]}' for t in df.columns[1:].to_series().str.join('').index.values]
    df.columns = ['object_id'] + flat_index


    types_dict = {feat_name_: np.float32 for feat_name_ in df.columns[1:]}
    types_dict['object_id'] = np.uint32
    df = df.astype(types_dict)


    df.reset_index(drop=True).to_hdf(save_dir+'/'+save_name, key='w')


set_str = 'training'
st = time.time()
main(
    save_dir='data/'+set_str+'_feats',
    save_name='cesium_full_feats.h5',
    light_curves_dir='data/'+set_str+'_cesium_curves',
    n_batches=2,
)
print(f'>   featgen_cesium : Wall time : {(time.time()-st)/60:.2f} minutes')