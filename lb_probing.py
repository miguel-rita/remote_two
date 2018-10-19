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

# Prob class 99 bin number k with probability p
probes = [0.1, 0.3, 0.5, 0.7]
for i, p in tqdm.tqdm(enumerate(probes), total=len(probes)):

    bin_num = 0 # Galactic class 99 bin

    probs = np.zeros((num_ids, num_classes))
    probs[rs_bins == bin_num, -1] = p # class 99 col
    probs[rs_bins != bin_num, -1] = 1-p  # class 99 col
    probs[:,-4] = 1 - probs[:,-1] # class 90 (-4) col
    sub = np.hstack([object_ids, probs])

    h = ''
    for s in col_names:
        h += s + ','
    h = h[:-1]

    # Write to file
    np.savetxt(
        fname='./subs/crazy_sauce_'+str(i)+'.csv',
        X=sub,
        fmt=['%d']+['%.1f']*num_classes,
        delimiter=',',
        header=h,
        comments='',
    )

print('Done . . .')