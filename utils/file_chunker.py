import numpy as np
import pandas as pd
import tqdm
import gc

'''
Chunking utils, fixed & fast
'''

def get_splits(oids, nsplits, consider_global_header=True):
    '''
    Given a numpy array containing sequential object ids, not necessarily sorted, will return via 'final_splits'
    the indexes where splitting occurs, each index indicating the first element of the chunk, as well as number of
    elements in chunk via 'nrows_per_split'
    :param consider_global_header bool used to add +1 to all split locations
    :param nsplits is the number of final data chunks
    '''
    rough_splits = np.round(np.linspace(0, oids.size, nsplits+1)[:-1], 0).astype(int)

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

    if consider_global_header:
        final_splits = [s + 1 for s in final_splits]  # +1 to jump global header

    nrows_per_split = [final_splits[i + 1] - final_splits[i] for i in np.arange(len(final_splits) - 1)]

    return final_splits, nrows_per_split

def main(src_str, chunk_size):

    # Configs
    save_dir = f'../data/{src_str}_chunks/'
    chunk_name = f'{src_str}_set_chunk'
    comp = '.zip' if src_str == 'test' else ''
    source_csv = f'../data/{src_str}_set.csv' + comp

    print('>    file_chunker : Reading total file size . . .')

    # # Small file solution
    # n_rows = np.sum([1 for _row in open(source_csv, 'r')])

    # Large file solution
    n_rows = np.load('../data/test_obj_ids.npy').size
    print(n_rows)

    n_chunks = np.ceil(n_rows/chunk_size)
    prev_chunk = pd.DataFrame()

    print('>    file_chunker : Getting iterator . . .')
    it = pd.read_csv(
        filepath_or_buffer=source_csv,
        chunksize=chunk_size,
        dtype={
            'object_id': np.int32,
            'mjd': np.float32,
            'passband': np.int8,
            'flux': np.float32,
            'flux_err': np.float32,
            'detected': np.uint8,
        },
        iterator=True
    )

    # # TEST
    # fulldf = pd.read_csv(
    #     filepath_or_buffer=source_csv,
    #     dtype={
    #         'object_id': np.int32,
    #         'mjd': np.float32,
    #         'passband': np.int8,
    #         'flux': np.float32,
    #         'flux_err': np.float32,
    #         'detected': np.uint8,
    #     },
    # )
    # dfs=[]

    print('>    file_chunker : Starting chunking process . . .')

    for i, curr_chunk in tqdm.tqdm(enumerate(it), total=n_chunks):

        if i == 0: # 'Skip' first iteration to always have 2 chunks loaded in mem at same time
            prev_chunk = curr_chunk
            continue

        last_prev_id = prev_chunk['object_id'].values[-1]
        curr_ids = curr_chunk['object_id']

        # Now with two chunks in memory move extra ids in curr chunk back to prev chunk
        # This way prev chunk will always be complete
        for num_chunks_to_move_up, oid in enumerate(curr_ids):
            if oid != last_prev_id:
                break
            last_prev_id = oid

        if num_chunks_to_move_up > 0: # Need border adjustment between chunks
            prev_chunk = pd.concat([prev_chunk, curr_chunk.iloc[:num_chunks_to_move_up, :]])
            curr_chunk = curr_chunk.iloc[num_chunks_to_move_up:, :]

        # Save prev_chunk
        prev_chunk.to_hdf(save_dir + chunk_name + str(i-1) + '.h5', key='w', mode='w') # i-1 since we skipped first it
        # # TEST
        # dfs.append(prev_chunk)

        # If last chunk lats save it too
        if i == n_chunks-1:
            curr_chunk.to_hdf(save_dir + chunk_name + str(i) + '.h5', key='w', mode='w') # i-1 since we skipped first it
            # # TEST
            # dfs.append(curr_chunk)
        prev_chunk = curr_chunk

        del curr_chunk
        gc.collect()

    # # TEST CODE
    # fulldf2 = pd.concat(dfs, axis=0)
    # assert np.array_equal(fulldf.values, fulldf2.values)

if __name__ == '__main__':
    main('test', chunk_size=np.ceil(453653104/32))
