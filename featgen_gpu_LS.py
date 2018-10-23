import pandas as pd
import numpy as np
import cuvarbase.lombscargle as gls
from utils.utils import load_lightcurves_from_path
import matplotlib.pyplot as plt
import tensorflow as tf
import os, time

def calc_LS_chunk(lc_list):
    '''
    Calculate LS list

    :param lc_list : list of n+1 light curves as [(ts0,vs0,es0), ..., (tsn,vsn,esn)]
    :return:
    '''

    os.environ['PATH'] = '/usr/local/cuda/bin'
    os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/cuda/lib'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib'
    os.environ['CUDA_ROOT'] = '/usr/local/cuda'

    st = time.time()
    print(f'>   calc_LS_chunk : Starting calc . . .')

    # Set up LombScargleAsyncProcess (compilation, etc.)
    proc = gls.LombScargleAsyncProcess()
    # Run on lightcurve batch
    results = proc.batched_run_const_nfreq(lc_list)
    # Synchronize all cuda streams
    proc.finish()

    et = time.time()
    print(f'>   calc_LS_chunk : Wall time of {et-st} secs')

    return results

def generate_LS_tensor(ls_list):
    '''
    Generate a 3D tensor from a list of equally-freq sampled lightcurve LSs

    :param lc_list:
    :return:
    '''

    n_passbands = 6
    n_oids = int(len(ls_list) / n_passbands)
    n_freqs = ls_list[0][0].size
    ls_tensor = np.zeros((n_oids, n_passbands, n_freqs))

    for oid in range(n_oids):
        oid_powers = np.vstack([lc[1] for lc in ls_list[oid*6:(oid+1)*6]])
        ls_tensor[oid, :, :] = oid_powers

    return ls_tensor.astype(np.float32)

def gpu_compute_ls_feats(ls_tensor, n_bins):
    '''
    TF computing of LS feats

    :param ls_tensor:
    :param n_bins : int, num of linear freq bins
    :return:
    '''

    def generate_ls_feats(ls_t, n_bins):
        '''
        Op. to compute LS feats per bin

        Implemented feats per bin:
            - Max

        :param n_bins : int, num of linear freq bins
        '''

        # Compute freq range splits
        freq_splits = [s.astype(np.int32) for s in np.array_split(np.arange(ls_t.shape[-1].value), n_bins)]

        single_feats = []
        feat_names = []
        for i,split in enumerate(freq_splits):

            ls_t_bin = ls_t[:, :, split[0]:split[-1]+1]

            single_feats.extend(
                [
                    #tf.reduce_mean(ls_t_bin, axis=[2]),
                    tf.reduce_max(ls_t_bin, axis=[2]),
                    #tf.reduce_min(ls_t_bin, axis=[2]),
                ]
            )
            for fname in ['max']:#['mean', 'max', 'min']:
                feat_names.extend([f'{fname}_ch{j}_bin{i}' for j in range(ls_t.shape[1].value)])

        return tf.concat(single_feats, axis=1), feat_names

    # Placeholder for ls tensor
    ls_t = tf.placeholder(dtype=tf.float32, shape=np.shape(ls_tensor))
    ls_feats, feat_names = generate_ls_feats(ls_t, n_bins=n_bins)

    # Run session
    with tf.Session() as sess:
        feats = sess.run(ls_feats, feed_dict={ls_t: ls_tensor})
        sess.close()

    return feats, feat_names

light_curves = load_lightcurves_from_path('data/training_group_lcs/training_curves_lcs.pkl')
results = calc_LS_chunk(light_curves)
ls_tensor = generate_LS_tensor(results)

ls_feats, feat_cols = gpu_compute_ls_feats(ls_tensor, n_bins=10)

# Save LS vanilla train feats
oids = np.load('data/training_group_lcs/training_curves_oids.npy')
oids = np.expand_dims(oids, axis=1)
pd.DataFrame(
    data=np.hstack([oids, ls_feats]),
    columns=['object_id'] + feat_cols,
).to_hdf('data/training_feats/LS_vanilla_train_feats_10bins.h5', 'w')

print('Done')





############
# Plotting #
############

# soid = 45
# eoid = soid+6
#
#
# f, axes = plt.subplots(6, 6,
#                        figsize=(12,12))
#
# for (frqs, ls_power), ax in zip(results[soid*6:eoid*6],
#                                       np.ravel(axes),
#                                       ):
#         ax.set_xscale('log')
#         ax.plot(frqs, ls_power)
#         #ax.axvline(freq, ls=':', color='r')
#         #ax.set_ylim([0, 0.8])
#
# f.text(0.05, 0.5, "Lomb-Scargle", rotation=90,
#        va='center', ha='right', fontsize=10)
# f.text(0.5, 0.05, "Frequency",
#        va='top', ha='center', fontsize=10)
#
#
#
# f.tight_layout()
# f.subplots_adjust(left=0.1, bottom=0.1)
# print(pd.read_csv('data/training_set_metadata.csv')['target'].iloc[soid:eoid])
# plt.show()

