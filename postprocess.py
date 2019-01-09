import pandas as pd
import numpy as np
import tqdm

'''
Postprocessing histogram shifting
'''

def build_prior_table(approx_freqs_dict):
    classes = np.sort(np.unique(np.array([int(s.split('_')[1]) for s in approx_freqs_dict.keys()])))
    lookup = {c: i for i, c in enumerate(classes)}

    ptable = np.zeros((classes.shape[0], 2))
    for k, v in approx_freqs_dict.items():
        class_ = int(k.split('_')[1])
        bin_str = k.split('_')[0]
        bin_index = 0 if bin_str == 'g' else 1
        row_index = lookup[class_]
        ptable[row_index, bin_index] = v

    # Up until here ptable denotes p(class=ci AND bin=gi). We must compute ptable p(class=ci | bin=gi). Thus:
    ptable /= np.sum(ptable, axis=0)

    return ptable


def build_sub_table(sub_, rs_bins_):
    return np.vstack([
        np.mean(sub_[rs_bins_ == 0, 1:], axis=0),
        np.mean(sub_[rs_bins_ != 0, 1:], axis=0),
    ]).T

def migrate_sub(prior_table_, sub_table_, sub__, rs_bins_):
    sub_ = np.copy(sub__)


    # Step 0 - Define mig. frontiers
    # mig_frontiers = np.array([
    #     [.95, .99],
    #     [.85, .95],
    #     [.9, .99],
    #     [.7, .95],
    #     [.7, .95],
    #     [.99, .999],
    #     [.9, .95],
    #     [.95, .99],
    #     [.98, .99],
    #     [.95, .99],
    #     [.96, .98],
    #     [.6, .95],
    #     [.92, .94],
    #     [.95, .99]
    # ])
    mig_frontiers = np.vstack([[0.5, 0.8] for i in range(14)])

    mig_frontiers = np.hstack([np.zeros((mig_frontiers.shape[0],1)), mig_frontiers, np.ones((mig_frontiers.shape[0],1))])

    # # DUMMY FRONTS
    # mig_frontiers = np.array([
    #     [0, .5, 1],
    #     [0, .5, 1],
    #     [0, .5, 1],
    #     [0, .5, 1],
    # ])

    # Step I - build migration matrix

    sub_set = sub_[:, 1:-1] # Discard object_id and class 99

    ideal_sub_set = np.zeros(sub_set.shape)

    # Precompute bin prob sums
    bin_sums_precomputed = [
        np.sum(sub_[rs_bins_==0,1:]), #Discard o_id but keep class 99
        np.sum(sub_[rs_bins_!=0,1:]), #Discard o_id but keep class 99
    ]

    for col_num in np.arange(sub_set.shape[1]):  # For each class except 99 determine const scale factor

        # Define prob bins
        cond_probs_bins = [
            sub_set[rs_bins_ == 0, col_num],
            sub_set[rs_bins_ != 0, col_num]
        ]

        for bin_n in [0,1]:

            cond_probs_bin = cond_probs_bins[bin_n]

            # Calc factor
            factor = prior_table_[col_num, bin_n] / sub_table_[col_num, bin_n]

            # Get sorted probs
            sort_by_probs = np.argsort(cond_probs_bin) # Ascending
            cond_probs_bin = cond_probs_bin[sort_by_probs]

            # Calculate sum borders
            num_elements = cond_probs_bin.shape[0]
            borders = np.floor(mig_frontiers[col_num,:] * num_elements).astype(int)

            # Calculate sums
            sums = np.array([np.sum(cond_probs_bin[borders[i]:borders[i+1]]) for i in range(borders.shape[0]-1)])

            # Define m weights
            m_w = np.array([0,0.5,1])
            # #DUMMY
            # m_w = np.array([0.5,1])

            # Calculate m depending on factor
            sum_bin = bin_sums_precomputed[bin_n]

            # Reverse weights if we need to decrease overall probability
            if factor < 1:
                m_w = m_w[::-1]

            # Calc m
            sum_inner = np.inner(m_w, sums)
            assert sum_inner != 0
            m = (sum_bin * prior_table_[col_num, bin_n] - np.sum(sums)) / sum_inner

            # Apply m to produce ideal sub
            ideal_cond_probs_bin = cond_probs_bin.copy()
            for i in range(borders.shape[0] - 1):
                ideal_cond_probs_bin[borders[i]:borders[i+1]] *= (1 + m_w[i] * m)

            # Unsort back
            ideal_cond_probs_bin = ideal_cond_probs_bin[np.argsort(sort_by_probs)]

            # Plug into ideal sub matrix
            if bin_n==0:
                ideal_sub_set[rs_bins_==0,col_num] = ideal_cond_probs_bin
            else:
                ideal_sub_set[rs_bins_!=0,col_num] = ideal_cond_probs_bin

    migration_matrix = sub_set - ideal_sub_set












    # Step II - migrate probabilities per row

    # For each submission row . . .
    for i, (sub_line, mm_line) in tqdm.tqdm(enumerate(zip(sub_[:,1:-1], migration_matrix)), total=sub_.shape[0]):

        # Get positive col indexes - classes where probs need to go down
        pos_cols = np.where(mm_line > 0)[0]

        # Get negative col indexes sorted by descending sub confidence
        # This makes sense since we'll transfer superavit probability elsewhere where we are more confident first
        neg_cols = np.where(mm_line < 0)[0]
        sorted_ixs = np.argsort(sub_line[neg_cols])[::-1]
        neg_cols = neg_cols[sorted_ixs]

        # For each positive col try to empty it across neg cols
        for pos_col in pos_cols:
            for neg_col in neg_cols:
                if mm_line[neg_col] == 0:  # Neg col already satisfied
                    continue

                budget = np.min([mm_line[pos_col], sub_line[pos_col]]) # To prevent taking more prob than we have

                max_necessity = -np.min([np.abs(mm_line[neg_col]), 1-sub_line[neg_col]])
                leftover_budget = budget + max_necessity  # Update budget

                if leftover_budget <= 0:
                    sub_line[pos_col] -= budget
                    sub_line[neg_col] += budget
                    mm_line[neg_col] += budget
                    break  # Budget just ran out, next pos col - must break out to pos cols loop
                else:
                    sub_line[pos_col] += max_necessity
                    mm_line[pos_col] += max_necessity # Update migration superavit
                    sub_line[neg_col] += -max_necessity  # - to get abs val
                    mm_line[neg_col] = 0  # Neg col satisfied

        # Assign migrated line
        sub_[i, 1:-1] = sub_line
        assert not np.any(sub_line<0)
        assert not np.any(sub_line>1)

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
test_prior_table = test_sub_table.copy()
test_prior_table[:2,:] += 0.1
test_prior_table[2:-1,:] -= 0.1

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
sub = pd.read_csv('subs/sub_v3_isband_0.7691.csv').values

# Load rs_bin info
rs_bins = np.load('data/rs_bins.npy')

prior_table = build_prior_table(approx_freqs)
sub_table = build_sub_table(sub, rs_bins)

np.set_printoptions(suppress=True)
print(prior_table)
print(sub_table)

shifted_sub = migrate_sub(prior_table, sub_table, sub, rs_bins)
print(build_sub_table(shifted_sub, rs_bins))

# Save shifted sub

# Get submission header
col_names = list(pd.read_csv(filepath_or_buffer='data/sample_submission.csv', nrows=1).columns)
num_classes = len(col_names) - 1

h = ''
for s in col_names:
    h += s + ','
h = h[:-1]

np.savetxt(
    fname='subs/sub_v3_isband_minorgatsby_0.7691.csv',
    X=shifted_sub,
    fmt=['%d'] + ['%.4f'] * num_classes,
    delimiter=',',
    header=h,
    comments='',
)