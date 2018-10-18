import numpy as np
import pandas as pd
import tqdm
import gc

'''
WARNING : pd skiprows doesn't avoid loading entire file into memory . Possible solution here :
        https://stackoverflow.com/questions/42228770/load-pandas-dataframe-with-chunksize-determined-by-column-variable
        
        Curr runtime for 16 chunks ~20mins
'''

def get_splits(oids, nsplits):
    '''
    Given a numpy array containing sequential object ids, not necessarily sorted, will return via 'final_splits'
    the indexes where splitting occurs, each index indicating the first element of the chunk, as well as number of
    elements in chunk via 'nrows_per_split'
    '''
    rough_splits = np.round(np.linspace(0, oids.size, nsplits)[:-1], 0).astype(int)

    # Refine splits
    final_splits = []
    for s in rough_splits:
        p_id = oids[s]
        n_id = oids[s]
        ind = s
        while n_id == p_id:
            p_id = n_id
            ind -= 1
            n_id = oids[ind]
        final_splits.append(ind + 1)
    final_splits.append(oids.size)  # Not size-1 to account for global  header

    final_splits = [s + 1 for s in final_splits]  # +1 to jump global header

    nrows_per_split = [final_splits[i + 1] - final_splits[i] for i in np.arange(len(final_splits) - 1)]

    return final_splits, nrows_per_split

# Load test object_ids. These are contiguous yet unsorted
print(f'\nLoading test object ids . . .')
toids = np.load('../data/test_obj_ids.npy')

# Get split points
nsplits = 16
splits, chunksizes = get_splits(toids, nsplits=nsplits)

# Split
save_dir = '../data/test_chunks/'
chunk_name = 'test_set_chunk'
source_csv = '../data/test_set.csv.zip'
header = ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected']

for i, (chunk_start_index, chunk_size) in tqdm.tqdm(enumerate(zip(splits, chunksizes)), total=nsplits):
    print(f'\nOpening chunk number {i} . . .')
    csv_chunk = pd.read_csv(
        source_csv,
        skiprows=chunk_start_index,
        nrows=chunk_size,
        header=None,
        names=header,
        dtype={
            'object_id' : np.int32,
            'mjd' : np.float32,
            'passband' : np.int8,
            'flux' : np.float32,
            'flux_err' : np.float32,
            'detected' : np.uint8,
        }
    )
    csv_chunk.to_hdf(save_dir+chunk_name+str(i)+'.h5', key='w', mode='w')
    del csv_chunk
    gc.collect()
