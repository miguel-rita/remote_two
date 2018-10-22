import pandas as pd
import numpy as np
import tqdm
import utils.preprocess as utils

# Get submission header
col_names = list(pd.read_csv(filepath_or_buffer='data/sample_submission.csv', nrows=1).columns)
num_classes = len(col_names) - 1

# Get test data
_train, test, _y_tgt, _train_cols = utils.prep_data()

# Get test ids
object_ids = test[['object_id']].values.astype(np.int32)
num_ids = object_ids.size

# RS bins
rs_bins = test['rs_bin'].values.astype(np.int32)

# Standard binary bin probing probabilities
P = [
    0.1,
    0.3,
    0.5,
    0.7,
]

# List of pair probes
pairs = [
    (99, 90),
    (95, 92),
    (88, 67),
    (65, 64),
    (62, 53),
    (52, 42),
    (52, 42),
    (16, 15),
    (99, 6),
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

for tgt_class, comp_class in tqdm.tqdm(pairs, total=len(pairs)):

    bin_num = 0  # Galactic bin

    tgt_class_col = classes.index(tgt_class) # Col of class being probed with prob p on bin_num, 1-p elsewhere
    comp_class_col = classes.index(comp_class) # Col of class being probed with prob 1-p on bin_num, p elsewhere

    for i, p in tqdm.tqdm(enumerate(P), total=len(P)):

        probs = np.zeros((num_ids, num_classes))
        probs[rs_bins == bin_num, tgt_class_col] = p
        probs[rs_bins != bin_num, tgt_class_col] = 1-p
        probs[:,comp_class_col] = 1 - probs[:,tgt_class_col]
        sub = np.hstack([object_ids, probs])

        h = ''
        for s in col_names:
            h += s + ','
        h = h[:-1]

        # Write to file
        np.savetxt(
            fname=f'./subs_freq_probe/c{tgt_class}_c{comp_class}_p{p:.2f}.csv',
            X=sub,
            fmt=['%d']+['%.1f']*num_classes,
            delimiter=',',
            header=h,
            comments='',
        )

print('Done . . .')