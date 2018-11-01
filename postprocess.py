import pandas as pd
import numpy as np
import tqdm

def build_prior_table(approx_freqs_dict):
    classes = np.sort(np.unique(np.array([int(s.split('_')[1]) for s in approx_freqs_dict.keys()])))
    print(classes)
    lookup = {c: i for i, c in enumerate(classes)}

    ptable = np.zeros((classes.shape[0], 2))
    for k, v in approx_freqs_dict.items():
        class_ = int(k.split('_')[1])
        bin_str = k.split('_')[0]
        bin_index = 0 if bin_str == 'g' else 1
        row_index = lookup[class_]
        ptable[row_index, bin_index] = v
    return ptable

def build_sub_table(sub_, rs_bins_):
    return np.vstack([
        np.sum(sub_[rs_bins_==0,1:], axis=0) / np.sum(sub_[:,1:]),
        np.sum(sub_[rs_bins_!=0,1:], axis=0) / np.sum(sub_[:,1:])
    ]).T

def migrate_sub(prior_table_, sub_table_, sub__, rs_bins_):
    sub_ = np.copy(sub__)
    # Step I - build migration matrix

    interp = 1.0  # % of density to migrate
    migration_matrix = np.zeros(sub_[:, 1:-1].shape)

    for col_num in np.arange(sub_[:, 1:-1].shape[1]):  # For each class except 99 determine const scale factor

        # Bin 0 - galactic
        migration_matrix[rs_bins_ == 0, col_num] = interp * prior_table_[col_num, 0] / sub_table_[col_num, 0]

        # Bins 1-9 - extragalactic
        migration_matrix[rs_bins_ != 0, col_num] = interp * prior_table_[col_num, 1] / sub_table_[col_num, 1]

    migration_matrix = sub_[:, 1:-1] * (1 - migration_matrix)

    # Step II - migrate probabilities per row

    # For each submission row . . .
    for i, (sub_line, mm_line) in tqdm.tqdm(enumerate(zip(sub_[:,1:-1], migration_matrix)), total=sub_.shape[0]):
        if i==7:
            print(1)
        # Get positive col indexes - classes where probs need to go down
        pos_cols = np.where(mm_line > 0)[0]

        # Get negative col indexes sorted by descending sub confidence
        # This makes sense since we'll transfer superavit probability elsewhere where we are more confident
        neg_cols = np.where(mm_line < 0)[0]
        sorted_ixs = np.argsort(sub_line[neg_cols])[::-1]
        neg_cols = neg_cols[sorted_ixs]

        # For each positive col try to empty it across neg cols
        for pos_col in pos_cols:
            budget = mm_line[pos_col]
            for neg_col in neg_cols:
                if mm_line[neg_col] == 0:  # Neg col already satisfied
                    continue

                budget += mm_line[neg_col]  # Update budget

                if budget <= 0:
                    sub_line[pos_col] -= mm_line[pos_col]
                    sub_line[neg_col] += mm_line[pos_col]
                    mm_line[neg_col] += mm_line[pos_col]
                    break  # Budget just ran out, next pos col - must break out to pos cols loop
                else:
                    sub_line[pos_col] += mm_line[neg_col]
                    sub_line[neg_col] += -mm_line[neg_col]  # - to get abs val
                    mm_line[pos_col] += mm_line[neg_col] # Update migration superavit
                    mm_line[neg_col] = 0  # Neg col satisfied

        # Assign migrated line
        sub_[i, 1:-1] = sub_line
        assert not np.any(sub_line<0)

    return sub_

test_sub = np.array(
    [
        [np.inf, .2, .3, .1, .3, .1],
        [np.inf, .3, .3, .2, .1, .1],
        [np.inf, .5, .1, .1, .2, .1],
        [np.inf, .1, .7, .1, .0, .1],
    ]
)

test_rs_bins = np.array([0,0,1,1])
test_sub_table = build_sub_table(test_sub, test_rs_bins)
test_prior_table = np.array(
    [
        [0.125-0.07, 0.15+0.07 ],
        [0.15-0.07 , 0.2+0.07  ],
        [0.075+0.02, 0.05-0.02 ],
        [0.1+0.02  , 0.05-0.02 ],
        [0.05 , 0.05 ]]
)
# Test code
#new_sub = migrate_sub(test_prior_table, test_sub_table, test_sub, test_rs_bins)

'''

Marginal distribution weight shifting

'''

approx_freqs = {
    'g_99' : 0.002350,
    'g_90' : 0.001841,
    'eg_99' : 0.091153,
    'eg_90' : 0.086091,
    'g_95' : 0.001949,
    'g_92' : 0.002357,
    'eg_95' : 0.090235,
    'eg_92' : 0.000005,
    'g_88' : 0.002583,
    'g_67' : 0.002014,
    'eg_88' : 0.082226,
    'eg_67' : 0.083775,
    'g_65' : 0.030996,
    'g_64' : 0.001612,
    'eg_65' : 0.003594,
    'eg_64' : 0.094024,
    'g_62' : 0.001978,
    'g_53' : 0.002321,
    'eg_62' : 0.090185,
    'eg_53' : 0.000003,
    'g_52' : 0.002201,
    'g_42' : 0.002337,
    'eg_52' : 0.083214,
    'eg_42' : 0.082840,
    'g_16' : 0.030782,
    'g_15' : 0.001621,
    'eg_16' : 0.003579,
    'eg_15' : 0.093990,
    'g_6' : 0.024859,
    'eg_6' : 0.003284,
}

# Load submission
sub = pd.read_csv('subs/sub_nn_const99_0.9355.csv').values

# Load rs_bin info
rs_bins = np.load('data/rs_bins.npy')

prior_table = build_prior_table(approx_freqs)
sub_table = build_sub_table(sub, rs_bins)
print(prior_table, sub_table)
shifted_sub = migrate_sub(prior_table, sub_table, sub, rs_bins)

# Save shifted sub

# Get submission header
col_names = list(pd.read_csv(filepath_or_buffer='data/sample_submission.csv', nrows=1).columns)
num_classes = len(col_names) - 1

h = ''
for s in col_names:
    h += s + ','
h = h[:-1]

np.savetxt(
    fname='subs/sub_nn_const99_0.9355_post.csv',
    X=shifted_sub,
    fmt=['%d'] + ['%.4f'] * num_classes,
    delimiter=',',
    header=h,
    comments='',
)