import pandas as pd
import numpy as np

# Get submission header
col_names = list(pd.read_csv(filepath_or_buffer='data/sample_submission.csv', nrows=1).columns)
num_classes = len(col_names) - 1

# Get test ids
object_ids = pd.read_csv(filepath_or_buffer='data/test_set_metadata.csv', usecols=['object_id']).values.astype(int)
num_ids = object_ids.size

# Naive sub
eq_prob = np.ones((num_ids, num_classes)) * 1/15
sub = np.hstack([object_ids, eq_prob])

h = ''
for s in col_names:
    h += s + ','
h = h[:-1]

# Write to file
np.savetxt(
    fname='secret_sauce.csv',
    X=sub,
    fmt=['%d']+['%.3f']*num_classes,
    delimiter=',',
    header=h,
    comments='',
)

print('Done? ...')