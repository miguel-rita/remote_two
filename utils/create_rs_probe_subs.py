import pandas as pd
import numpy as np
import tqdm, os
import utils.preprocess as utils

'''
This file is used to generate 20 sub files to probe class 99 and 15 cardinality for each of
10 rs bins. Class 15 chosen as companion class due to high cardinality
'''

# Get submission header
col_names = list(pd.read_csv(filepath_or_buffer='../data/sample_submission.csv', nrows=1).columns)
num_classes = len(col_names) - 1

# Get test data
os.chdir('..')
_train, test, _y_tgt, _train_cols = utils.prep_data()

# Get test ids
object_ids = test[['object_id']].values.astype(np.int32)
num_ids = object_ids.size

# RS bins
rs_bins = test['rs_bin'].values.astype(np.int32)

# Standard bin probing probabilities
P = [
    0.2,
    0.8,
]

# Class vector
classes = [
    99,
    95,
    92,
    90,
    88,
    67,
    65,
    64,
    62,
    53,
    52,
    42,
    16,
    15,
    6,
]
classes = classes[::-1]

num_bins = 10
for bin_num in tqdm.tqdm(np.arange(num_bins), total=num_bins):

    class_99_col = classes.index(99) # Col of class being probed with prob p on bin_num, 1-p elsewhere
    class_15_col = classes.index(15) # Col of class being probed with prob 1-p on bin_num, p elsewhere

    for i, p in tqdm.tqdm(enumerate(P), total=len(P)):

        probs = np.zeros((num_ids, num_classes))
        probs[rs_bins == bin_num, class_99_col] = p
        probs[rs_bins != bin_num, class_99_col] = 1-p
        probs[:,class_15_col] = 1 - probs[:,class_99_col]
        sub = np.hstack([object_ids, probs])

        h = ''
        for s in col_names:
            h += s + ','
        h = h[:-1]

        # Write to file
        np.savetxt(
            fname=f'./subs_rs_probe/b{bin_num}_p{p:.1f}.csv',
            X=sub,
            fmt=['%d']+['%.1f']*num_classes,
            delimiter=',',
            header=h,
            comments='',
        )