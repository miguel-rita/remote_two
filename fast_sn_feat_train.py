import pandas as pd
import seaborn as sns
import tqdm, pickle
import numpy as np
import glob, pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize

'''
Fast prototype to test usage of feats derived from SN fits
'''

# Load metadata and target/pred info
meta = pd.read_hdf('data/train_small_subset.h5')
meta['target'] = np.load('data/target_col.npy')

# Check sn fit params
sn_fits = np.load('data/train_sn_fits.npy')
sn_fits = np.concatenate([sn_fits, np.tile(meta['target'].values, (6,1)).T[:,:,None]], axis=2)


# SAVE BEST T0s
#tfall_data = sn_fits[:,3:6,3]

# Maximum admissible error for best t0 estimate :
MAX_ADMISS_ERROR = 1500

# Grab t0 estimates and msqe info
t0_data = sn_fits[:,2:6,1]
msqe_info = sn_fits[:,2:6,4]
valid_rows_mask = np.logical_not(np.all(np.isnan(t0_data), axis=1))

# Placeholder - will have nans at least on rows that had 3 nans
best_t0 = t0_data[:,[0]]
masked_best_t0 = best_t0[valid_rows_mask]
arg_min_msqe = np.nanargmin(msqe_info[valid_rows_mask], axis=1)
for i, (t_line, argmin) in enumerate(zip(t0_data[valid_rows_mask], arg_min_msqe)):
    masked_best_t0[i,0] = t_line[argmin]
best_t0[valid_rows_mask] = masked_best_t0

data_arr = np.concatenate([meta['object_id'].values[:,None], best_t0], axis=1)
feat_names = ['best_t0']
# Build feat dataframe
feat_arr = pd.DataFrame(data=data_arr, columns=['object_id']+feat_names, index=None)
feat_arr = feat_arr.astype({'object_id' : np.uint64})
# Save df
feat_arr.to_hdf('data/training_feats/experimental_sn_v3_t0.h5', key='w')
# Save feat names
with open('data/training_feats/experimental_sn_v3_t0.pkl', 'wb') as handle:
    pickle.dump(list(feat_arr.columns)[1:], handle, protocol=pickle.HIGHEST_PROTOCOL)


def model(t, params):
    a0, t0, trise, tfall = params

    rise_exponent = -(t - t0) / trise
    fall_exponent = -(t - t0) / tfall

    res = a0 * np.e ** fall_exponent / (1 + np.e ** rise_exponent)

    return res


# Fast prototyping of sn train feats
bands = [3]
oids = meta['object_id'].values
distmods = meta['distmod'].values
sn_feats = np.zeros((oids.size, len(bands) * 2))  # 2 feats per band : peak bright. and deltam15
sn_feats.fill(np.nan)

base_feats = ['peak_bright', 'delta15']
feat_names = []
for band in bands:
    feat_names.extend([f'{bf}_band{band:d}' for bf in base_feats])

for i, (oid, distmod) in enumerate(zip(oids, distmods)):
    for j, band_num in enumerate(bands):

        MAX_ADMISS_ERROR = 1500
        msqe = sn_fits[i, band_num, -2]
        if distmod == np.nan or msqe > MAX_ADMISS_ERROR:  # Intragal
            sn_feats[i, j * 2:j * 2 + 2] = np.array([np.nan, np.nan])
            continue

        # Get model params
        a0, t0, trise, tfall = sn_fits[i, band_num, 0], sn_fits[i, band_num, 1], sn_fits[i, band_num, 2], sn_fits[
            i, band_num, 3]

        # Calc t of peak brightness
        t_max = trise * np.log((tfall - trise) / trise) + t0

        # Calc flux of peak brightness and +15day peak brightness
        params = (a0, t0, trise, tfall)
        fluxes = model(np.array([t_max, t_max + 15]), params)

        # Convert to magnitudes
        abs_mags = -2.5*np.log10(fluxes) - distmod
        peak_bright = abs_mags[0]
        delta15 = abs_mags[1] - abs_mags[0]

        sn_feats[i, j * 2:j * 2 + 2] = np.array([peak_bright, delta15])

data_arr = np.concatenate([meta['object_id'].values[:,None], sn_feats], axis=1)
# Build feat dataframe
feat_arr = pd.DataFrame(data=data_arr, columns=['object_id']+feat_names, index=None)
feat_arr = feat_arr.astype({'object_id' : np.uint64})

# Save df
feat_arr.to_hdf('data/training_feats/experimental_sn_v1.h5', key='w')
# Save feat names
with open('data/training_feats/experimental_sn_v1.pkl', 'wb') as handle:
    pickle.dump(list(feat_arr.columns)[1:], handle, protocol=pickle.HIGHEST_PROTOCOL)
