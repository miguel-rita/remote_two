import time, tqdm
import dask.dataframe as dd
from utils.file_chunker import get_splits
st = time.time()

'''
TODO : NOT WORKING FOR LARGE SETS
'''
set_str= 'test'
df = dd.read_csv('data/'+set_str+'_set.csv', dtype={'object_id' : 'uint32', 'detected' : 'uint8'})

# Setup
chunk_dir = './data/'+set_str+'_chunks/'

print('>    dask_chunker : Setting index . . .')
df.set_index('object_id')

print('>    dask_chunker : Compute splits . . .')
oids = df['object_id'].compute().values
splits, _nrows = get_splits(oids, 8, consider_global_header=False)
print(splits)

# Fill dummy divisions, repartition below
df.divisions = tuple([0]*(len(df.divisions)-1)+[len(oids)])
df = df.repartition(divisions=splits, force=True)

print('>    dask_chunker : Save chunks . . .')
for i in tqdm.tqdm(range(df.npartitions), total=df.npartitions):
    p = df.partitions[i].compute()
    p.to_hdf(chunk_dir + set_str + f'_set_chunk{i:d}.h5', key='w', mode='w')

print('>    dask_chunker : Total time in s :', time.time()-st)