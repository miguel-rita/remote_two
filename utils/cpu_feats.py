import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, os, time, gc
from scipy.stats import kurtosis, skew
from utils.file_chunker import get_splits

# Atomic pool worker
def compute_subchunk_feats(sub_ck):

    sub_chunk_feats = pd.pivot_table(
        data=sub_ck,
        values=['flux', 'flux_err'],  # , 'mjd'],
        index=['object_id'],
        columns=['passband'],
        aggfunc={
            'flux': [np.mean, np.max, np.min, np.std, kurtosis, skew],
            'flux_err' : [np.mean],
            #'mjd' : [np.ptp],
        }
    )

    return sub_chunk_feats

def compute_chunk_feats(ck, multiprocess = True):

    chunk_feats = pd.DataFrame()

    if multiprocess:
        # Get chunk splits for multiprocess

        oids = ck['object_id'].values
        splits, num_rows = get_splits(oids=oids, nsplits=mp.cpu_count(), consider_global_header=False)

        sub_cks = []
        for split, num_rows_ in zip(splits, num_rows):
            sub_cks.append(ck.iloc[split:split + num_rows_, :])

        # Dispatch work to processes

        pool = mp.Pool(processes=mp.cpu_count())
        sub_cks_feats = pool.map(compute_subchunk_feats, sub_cks)

        # Gather process work together

        chunk_feats = pd.concat(sub_cks_feats, axis=0)
        del sub_cks_feats

    else: # Single-process
        chunk_feats = compute_subchunk_feats(ck)

    # Convert to float16 and flatten multiindex
    cns = [f'{t[0]}_{t[1]}_{t[2]}' for t in chunk_feats.columns.to_series().str.join('').index.values]
    chunk_feats = chunk_feats.astype(
        {feat_name_: np.float16 for feat_name_ in chunk_feats}
    )
    chunk_feats.columns = pd.Index(cns)

    return chunk_feats

def compute_feats_cpu():

    st = time.time()
    print(f'>   cpu_feats : Started . . .')

    # Get existing chunk names
    s = 'test'
    chunks_dir = '../data/' + s + '_chunks/'
    feats_dir = '../data/' + s + '_feats/'
    feats_name = s + '_set_feats_v4.h5'
    chunk_names = [n for n in os.listdir(chunks_dir) if n[0]=='t']

    feats_df = pd.DataFrame()

    for i, chunk_name in tqdm.tqdm(enumerate(chunk_names), len(chunk_names)):

        cst = time.time()
        print(f'>   cpu_feats : Working on chunk number {i} . . .')

        ck = pd.read_hdf(chunks_dir + chunk_name)

        chunk_feats = compute_chunk_feats(ck, multiprocess=True)

        feats_df = pd.concat([feats_df, chunk_feats], axis=0)
        del ck, chunk_feats
        gc.collect()
        print(f'>   cpu_feats : Done with chunk {i} in {time.time()-cst:.2f} seconds')

    feats_df = feats_df.reset_index().sort_values('object_id').reset_index(drop=True)
    feats_df.to_hdf(feats_dir + feats_name, key='w', mode='w')
    print(f'>   cpu_feats : Done in {time.time()-st:.3f} seconds')

if __name__ == '__main__':
    compute_feats_cpu()


