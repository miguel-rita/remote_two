import numpy as np
import pandas as pd
import multiprocessing
import tqdm, os, time, gc
from scipy.stats import kurtosis, skew
from utils.chunk_test_set import get_splits

# Get existing chunk names
s = 'train'
chunks_dir = '../data/'+s+'_chunks/'
feats_dir = '../data/'+s+'_feats/'
feats_name = s+'_set_feats_v2.h5'

cnames = [n for n in os.listdir(chunks_dir) if n != 'dir.txt']

# Compute feats per chunk
feats_df = pd.DataFrame()
mt = time.time()
for cname in cnames:

    st = time.time()
    ck = pd.read_hdf(chunks_dir + cname)
    print(f'>   cpu_feats : Loaded chunk in {time.time()-st:.3f} seconds')



    # Split chunk for multicore processing

    # Create feats
    st = time.time()

    oids = ck['object_id'].values



    cfeats = pd.pivot_table(
        data = ck,
        values = ['flux'],#, 'mjd'],
        index = ['object_id'],
        columns = ['passband'],
        aggfunc = {
            'flux' : [np.mean, np.max, np.min, np.std, kurtosis, skew],
            #'mjd' : [np.ptp],
        }
    )




    # Convert to float16 and flatten multiindex
    cns = [f'{t[0]}_{t[1]}_{t[2]}' for t in cfeats.columns.to_series().str.join('').index.values]
    cfeats = cfeats.astype(
        {feat_name_ : np.float16 for feat_name_ in cfeats}
    )
    cfeats.columns = pd.Index(cns)
    print(f'>   cpu_feats : Created chunk feats in {time.time()-st:.3f} seconds')

    feats_df = pd.concat([feats_df, cfeats], axis=0)
    del ck, cfeats
    gc.collect()

feats_df = feats_df.reset_index().sort_values('object_id').reset_index(drop=True)
feats_df.to_hdf(feats_dir + feats_name, key='w', mode='w')
print(f'>   cpu_feats : Done in {time.time()-mt:.3f} seconds')




