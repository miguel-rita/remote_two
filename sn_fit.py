import pandas as pd
import seaborn as sns
import tqdm
import numpy as np
import glob, pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize

'''
Function defs
'''
def model(t, params):
    a0, t0, trise, tfall = params

    rise_exponent = -(t - t0) / trise
    fall_exponent = -(t - t0) / tfall

    res = a0 * np.e ** fall_exponent / (1 + np.e ** rise_exponent)

    return res

def msqe(x0, *args):
    tds, mds = args
    return np.sum((model(tds, x0) - mds) ** 2)

'''
Data loading
'''

# Load train set on cesium format
cs_ = glob.glob('data/training_cesium_curves/*.pkl')
oids_ = glob.glob('data/training_cesium_curves/*.npy')

# Sort correctly
_cs2 = cs_[1:6].copy()
_oids2 = oids_[1:6].copy()
del cs_[1:6], oids_[1:6]
cs_.extend(_cs2)
oids_.extend(_oids2)

ts_a, ms_a, es_a, ds_a, oids_a = [], [], [], [], []
for c_, o_ in zip(cs_, oids_):
    with open(c_, 'rb') as fp:
        full_c=pickle.load(fp)
    oids_a.append(np.load(o_))
    ts_a.extend(full_c[0])
    ms_a.extend(full_c[1])
    es_a.extend(full_c[2])
    ds_a.extend(full_c[3])
oids_a = np.hstack(oids_a)

# Load metadata and target/pred info
meta = pd.read_hdf('data/train_small_subset.h5')
meta['target'] = np.load('data/target_col.npy')

'''
Configuration
'''

# Define output vars
sn_fits = np.zeros((oids_a.size, 6, 5)) # 6 bands, 4 model params (a0, t0, trise, tfall) + msqe
sn_fits.fill(np.nan)

# Select sn class
target = 'all'
distmods = meta['distmod'].values

if target != 'all':
    sn_oids = meta.loc[(meta['target'] == target), 'object_id'].values
    sn_oids = np.array([56821])
    mask = np.arange(0, len(ts_a))[np.isin(oids_a, sn_oids)]
    ts = [ts_a[i] for i in mask]
    ms = [ms_a[i] for i in mask]
    es = [es_a[i] for i in mask]
    ds = [ds_a[i] for i in mask]
    oids = oids_a[mask]
else:
    ts, ms, es, ds, oids = ts_a, ms_a, es_a, ds_a, oids_a
N_CURVES = len(ts)

# Plt confs
PLOT_ON = False

if PLOT_ON:
    N_CURVES = 40  # Num. curves to plot
    plotspace_x = np.linspace(59000, 62000, 1000) # Time plot space
    plt.close('all')
    plt.cla()
    plt.clf()
    f, axs = plt.subplots(N_CURVES, 1, sharex=False, sharey=False, figsize=(30, N_CURVES * 4))
    colors = ['darkorchid', 'royalblue', 'forestgreen', 'orange', 'indianred', 'peru']

# Loop collectors
msqes = []

'''
Fit and (optionally) plot curves
'''

for j, (tt, mm, ee, dd, oidd, distmod) in tqdm.tqdm(enumerate(
        zip(ts[:N_CURVES], ms[:N_CURVES], es[:N_CURVES], ds[:N_CURVES], oids[:N_CURVES], distmods[:N_CURVES])), total=len(ts)):

    det_count = np.zeros(6)
    ctypes_log = []
    for band_num, (t, m, e, d) in enumerate(zip(tt, mm, ee, dd)):

        if band_num not in [0,1,2,3,4,5]:
            continue

        arg_m_max = np.argmax(m * d)
        m_max, t_max = m[arg_m_max], t[arg_m_max]

        det_count[band_num] = np.sum(d)
        if det_count[band_num] > 1:

            # 1. Get curve type
            dflag = d.astype(bool)

            if not np.any(m[dflag] > m[dflag][0]):
                curve_type = 'tail'
            elif not np.any(m[dflag] > m[dflag][-1]):
                curve_type = 'head'
            else:
                curve_type = 'peak'

            # Force tails disguised as peaks - we know they're tails due to magnitude near noise level
            if curve_type == 'peak':
                ORDER_OF_MAG_CONST = 5
                if np.sum(d) < d.size: # If we have non detected points too (0s besides 1s) - percentile to handle noise outliers
                    if np.max(m[dflag]) < ORDER_OF_MAG_CONST * np.percentile(m[np.logical_not(dflag)], 85):
                        curve_type = 'tail'

            ctypes_log.append(curve_type)

            # 2. Build gap_matrix, indicating where measurement period start/end, per row

            GAP_THRESH = 100  # Empirical

            # Calculate points where cadence observation gaps start/end
            gaps = t[1:] - t[:-1]
            gaps[0], gaps[-1] = False, False # Ignore 1-hit obs intervals at time start/end

            gaps = np.hstack([gaps[0], gaps])
            gaps = gaps > 100

            # Ignore 1-hit obs intervals time middle
            flag_single_obs_ints = np.logical_and(gaps[1:], gaps[:-1])
            if np.any(flag_single_obs_ints):
                fixed_col = gaps[1:]
                fixed_col[flag_single_obs_ints] = False
                gaps[1:] = fixed_col

            gaps = np.logical_or(gaps, np.roll(gaps, len(gaps) - 1))
            # Mark beginning of first observations and end of last
            gaps[0], gaps[-1] = True, True
            gap_matrix = np.reshape(t[gaps], (-1,2)) # 2 cols, for beg and end of gap

            # 3. Determine t0 params : t0, left bound and right bound

            SN_DURATION = 200 # Empirical order of mag, constant
            lbs, rbs = np.nan, np.nan

            if curve_type == 'tail':
                # Get end of previous obs interval as lbs
                for i, interval in enumerate(gap_matrix):
                    if t_max<=interval[1] and t_max>=interval[0]: # Found t_max obs interval
                        if i == 0: # Edge case - tail on first obs period
                            lbs, rbs = t_max - SN_DURATION, t_max
                            t0 = (lbs + rbs) / 2
                        else:
                            lbs, rbs = gap_matrix[i-1, 1], t_max
                            t0 = (lbs + rbs)/2
                        a0 = 5 * m_max
                        break
            elif curve_type == 'head':
                # Get beg of next obs interval as rbs
                for i, interval in enumerate(gap_matrix):
                    if t_max <= interval[1] and t_max >= interval[0]:  # Found t_max obs interval
                        if i == gap_matrix.shape[0]-1: # Edge case - head on last obs period
                            lbs, rbs = t_max, t_max + SN_DURATION
                            # t0 = (lbs + rbs) / 2
                        else:
                            lbs, rbs = t_max + 5, gap_matrix[i + 1, 0]
                            # t0 = (lbs + rbs) / 2
                        t0 = t_max + 25
                        a0 = 3 * m_max
                        break
            else:
                assert curve_type == 'peak'

                # Constrain peak max within detection limits
                for di, ti in zip(d[arg_m_max:], t[arg_m_max:]):
                    if di == 1:
                        rbs = ti
                    else:
                        break
                for di, ti in zip(d[:arg_m_max+1][::-1], t[:arg_m_max+1][::-1]):
                    if di == 1:
                        lbs = ti
                    else:
                        break

                t0 = (lbs + rbs) / 2
                a0 = 2*m_max

            # 4. Fit curve

            trise = 2
            tfall = 20
            x0 = [a0, t0, trise, tfall]
            bnds = ((m_max/4, 800000), (lbs, rbs), (0.1, 5), (6,60))
            args = (t[d.astype(bool)], m[d.astype(bool)])
            res = minimize(msqe, x0, args, bounds=bnds)
            msqerror = msqe(res.x, *args)
            msqes.append(msqerror)

            # 4.5 Store fit data for later feat gen
            sn_fits[j,band_num,:4] = res.x
            sn_fits[j,band_num, 4] = msqerror

            # 5. Plotting

            if PLOT_ON:
                # Detection mask
                s = 20
                size_m = np.ones(len(d)) * s
                size_m[np.logical_not(d.astype(bool))] *= .1

                axs[j].scatter(t, m, s=size_m, c=colors[band_num], alpha=0.65, marker=None, cmap=None, norm=None, vmin=None,
                               vmax=None)
                axs[j].errorbar(t[d.astype(bool)], m[d.astype(bool)], yerr=e[d.astype(bool)], fmt='.', c=colors[band_num],
                                alpha=0.6)

                y_fit = model(plotspace_x, res.x)
                axs[j].scatter(plotspace_x, y_fit, s=6, c=colors[band_num], marker='x', alpha=0.3)

    rs = meta.loc[meta['object_id'] == oidd, 'hostgal_photoz'].values
    ddf = meta.loc[meta['object_id'] == oidd, 'ddf'].values

    if PLOT_ON:
        axs[j].set_title(
f'Class/OID : {target:d} {oidd:d} | Costs {sn_fits[j,3:6, 4]} | a1 : {res.x[0]:.0f} (a0 : {a0:.0f}) | \
t1 : {sn_fits[j,3:6, 1]} (t0 : {t0:.0f}) | trise1 : {sn_fits[j,3:6, 2]} (trise0 : {trise:.2f}) | \
tfall1 : {sn_fits[j,3:6, 3]} (tfall0 : {tfall:.2f}) | Type : {ctypes_log}'
        )

    # axs[j].set_ylim(-0.2,1.2)
    # axs[j].set_xlim(-500, 500)

if target=='all':
    print('> sn_fit :   Saving sn fit params array . . .')
    np.save('data/train_sn_fits.npy', sn_fits)
    print('> sn_fit :   . . . Finished saving sn fit params array.')

msqes = np.array(msqes)
print(f'MSQE : {np.mean(msqes):.2f}')

# sns.distplot(msqes[msqes<=4000], kde=False, bins=np.linspace(0,4000,50))
# plt.show()

if PLOT_ON:
    plt.savefig('edas/temp_sn_fit.png')
