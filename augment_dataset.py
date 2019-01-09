import numpy as np
import pandas as pd
import glob, pickle, tqdm

'''
Dataset augmentation-related utilities
'''

def build_oid_table(oids, tgts, weights_dict):
    '''
    Return a vector with the number of times to augment each oid
    
    :param oids: original oids
    :param tgts: target classes
    :param weights_dict: dict in format {class_num : multiple} where data will be augmented 'multiple' times
    :return: 1D array of size oids.size
    '''

    oids_table = np.zeros(oids.size)

    for k,v in weights_dict.items():

        # Mask only current class
        tgt_mask = tgts == k
        n_tgts = np.sum(tgt_mask)

        # Integer part of multiple
        int_mult = oids_table[tgt_mask]
        int_mult[:] = np.floor(v)

        # Spread remainder randomly through class samples
        n_extra = np.round((n_tgts * v) % n_tgts).astype(int)
        int_mult[np.random.choice(n_tgts, n_extra, replace=False)] += 1

        # Assign result
        oids_table[tgt_mask] = int_mult

    return oids_table.astype(int)

def augment_single_lc(ms, ts, es, ds, sample_freq=1.0):
    '''
    Augment a single (generally 6-band) light curve, in its time, flux, flux_err and detected dimensions
    
    :param ms, ts, es, ds: Source light curve, in cesium save format (4 parts)
    :param sample_freq: Augmented curve will have 'sample_freq' * n_orig points, where n_orig = number of original pts
    :return: Single augmented light curve m,t,e,d tuple
    '''

    new_ms, new_ts, new_es, new_ds = [], [], [], []

    for m,t,e,d in zip(ms, ts, es, ds):

        # Band matrix
        band_mat = np.vstack([m,t,e,d]).T

        # Compute detected groups
        prev_detected = False
        mjd_groups = []  # Final var holding all collect groups
        curr_group = []  # Temp var to collect group info in loop

        for line in band_mat:
            flux, mjd, ferr, detected = line[0], line[1], line[2], line[3]
            if prev_detected and not detected:
                # Just finished group
                mjd_groups.append(np.array(curr_group))
                curr_group = []
                prev_detected = False
            if detected:
                # Going through/starting group
                curr_group.append([flux, mjd, ferr, detected])
                prev_detected = True

        # Append last group
        if curr_group:
            mjd_groups.append(np.array(curr_group))

        # Split groups where two mjds are more than THRESHOLD apart
        THRESHOLD = 50
        new_mjd_groups = []
        for gp in mjd_groups:
            mjds = np.array([e[1] for e in gp]) # Get times

            if mjds.size == 1: # can't split a 1-element group
                new_mjd_groups.append(gp)
                continue

            ixs_to_split = mjds[1:]-mjds[:-1] > THRESHOLD # Get split points
            ixs_to_split = np.arange(1, mjds.size)[ixs_to_split]
            new_gps_ixs = np.split(np.arange(mjds.size), ixs_to_split) # Get split gp indexes
            for gp_ixs in new_gps_ixs:
                new_mjd_groups.append(np.array([gp[ix] for ix in gp_ixs]))
        mjd_groups = new_mjd_groups

        # Augment detected groups
        aug_grps = []
        for grp in mjd_groups:

            m_, t_, e_, d_ = grp[:,0], grp[:,1], grp[:,2], grp[:,3]
            n_random_pts = np.round(sample_freq * grp.shape[0]).astype(int)

            grp_t = np.random.uniform(np.min(t_), np.max(t_), size=n_random_pts)
            grp_m = np.interp(grp_t, t_, m_)
            grp_e = np.interp(grp_t, t_, e_)
            grp_d = np.round(np.interp(grp_t, t_, d_)).astype(np.int8) # Redudant but ok

            aug_grps.append(np.vstack([grp_m, grp_t, grp_e, grp_d]).T)

        grps_stack = np.vstack([*aug_grps, band_mat[d==0]])
        grps_stack = grps_stack[grps_stack[:,1].argsort()]

        new_m = grps_stack[:, 0]
        new_t = grps_stack[:, 1]
        new_e = grps_stack[:, 2]
        new_d = grps_stack[:, 3]

        new_ms.append(new_m)
        new_ts.append(new_t)
        new_es.append(new_e)
        new_ds.append(new_d)

    return new_ms, new_ts, new_es, new_ds

def augment_all_light_curves(lc_dir, oids_table):
    '''
    Augment all light curves in given dir 'lc_dir', according to frequencies in 'oids_table'
    
    :param lc_dir: Directory containing the source curves in cesium format
    :param oids_table: Vector containing number of augmentations per oid
    :return: List of augmented curves, including original curves, SORTED BY OID (important to align with aug metadata)
    '''

    oids, lcs_m, lcs_t, lcs_e, lcs_d = [], [], [], [], []

    lcs_paths = glob.glob(lc_dir + '/*.pkl')
    oids_paths = glob.glob(lc_dir + '/*.npy')
    lcs_paths.sort()
    oids_paths.sort()
    for lcs_path, oids_path in zip(lcs_paths, oids_paths):

        # Open curves
        with open(lcs_path, 'rb') as handle:
            tss, mss, ess, dss = pickle.load(handle)
            lcs_m.extend(mss)
            lcs_t.extend(tss)
            lcs_e.extend(ess)
            lcs_d.extend(dss)

        # Open oids
        oids.extend(np.load(oids_path))

    oids = np.array(oids) # List to np array

    aug_mss, aug_tss, aug_ess, aug_dss = [], [], [], []

    # Oids_table is sorted yet loaded oids and curves are not. Thus we'll unsort oids_table
    # to correctly augment the curves
    argsort = np.argsort(oids)
    oids_table = oids_table[np.argsort(argsort)]

    for ms, ts, es, ds, oid, freq in tqdm.tqdm(zip(lcs_m, lcs_t, lcs_e, lcs_d, oids, oids_table), total=len(lcs_m)):

        aug_mss.append(ms)
        aug_tss.append(ts)
        aug_ess.append(es)
        aug_dss.append(ds)

        for it in range(freq - 1):

            aug_ms, aug_ts, aug_es, aug_ds = augment_single_lc(ms, ts, es, ds, sample_freq=1.0)
            aug_mss.append(aug_ms)
            aug_tss.append(aug_ts)
            aug_ess.append(aug_es)
            aug_dss.append(aug_ds)

    # Sort by ascending oid to later match augmented metadata
    # NOTE ORDER : T, M, E, D, not M, T, E, D

    # Repeat unsorted oids using the already unsorted oids_table
    final_augs = []
    oids = np.repeat(oids, oids_table).astype(np.uint64)
    argsort = np.argsort(oids)
    for aug in [aug_tss, aug_mss, aug_ess, aug_dss]:
        aug = [aug[i] for i in argsort]
        final_augs.append(aug)

    return final_augs

def augment_meta(meta_df, oids_table, AUG_CONST=1e9):
    '''
    Augment metadata csv, according to freqs in 'oids_table'
    Note that all augmented oids are calculated as original oid * 'AUG_CONST' + n,
    where n=0 for the original oids, n=1 for the first augmentation, n=2 for the second . . .
    Note also that 'AUG_CONST' is > than every single original oid
    
    :param meta_df: pandas meta_df, direct from original .csv
    :param oids_table: Vector with augmentation frequencies per sample
    :return: augmented pandas df
    '''

    oids = meta_df['object_id'].values

    # Calculate aug oids values
    aug_oids = np.repeat(oids, oids_table).astype(np.uint64)
    prev = np.nan
    for i, aoid in enumerate(aug_oids):
        curr = aoid
        if curr == prev:
            aug_oids[i] = aug_oids[i-1] + np.array([1]).astype(np.uint64)
        else:
            aug_oids[i] *= AUG_CONST
        prev = curr

    aug_meta_df = meta_df.iloc[np.repeat(np.arange(oids.size), oids_table)]
    aug_meta_df.columns = meta_df.columns
    aug_meta_df.loc[:,'object_id'] = aug_oids

    return aug_meta_df

def main():

    # Load original metadata
    meta = pd.read_csv('data/training_set_metadata.csv')
    oids = meta['object_id'].values
    tgts = meta['target'].values

    unique, counts = np.unique(tgts, return_counts=True)
    counts = np.max(counts) / counts
    weights_dict = dict(zip(unique, counts))

    classes_to_keep = [42,52,62,67,90]
    for k in weights_dict.keys():
        if k not in classes_to_keep:
            weights_dict[k] = 1

    # Get aug frequencies
    # weights_dict = {
    #         95: 1,
    #         92: 1,
    #         90: 1,
    #         88: 1,
    #         67: 10,
    #         65: 1,
    #         64: 1,
    #         62: 4,
    #         53: 1,
    #         52: 10,
    #         42: 2,
    #         16: 1,
    #         15: 1,
    #         6: 1,
    #     }

    # Augment metadata
    oids_table = build_oid_table(oids=oids, tgts=tgts, weights_dict=weights_dict)
    aug_meta = augment_meta(meta_df=meta, oids_table=oids_table)

    # Save augmented metadata
    aug_meta.to_csv('data/aug_training_set_metadata.csv', index=False)

    # Augment lightcurve data
    lc_dir = 'data/training_cesium_curves'
    aug_lcs = augment_all_light_curves(lc_dir=lc_dir, oids_table=oids_table)

    # Save augmented lightcurve and oids data
    save_dir = 'data/aug_training_cesium_curves'

    with open(save_dir + '/' + 'aug_train_lcs.pkl', 'wb') as handle:
        pickle.dump(aug_lcs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(save_dir + '/' + 'aug_train_oids.npy', aug_meta['object_id'].values)

if __name__ == '__main__':
    main()





