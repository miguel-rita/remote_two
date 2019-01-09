'''
Determine class 99 and 15 frequencies per rs bin after probing
'''

import numpy as np
from scipy.optimize import least_squares

# Standard binary bin probing probabilities
P = [
    0.2,
    0.8,
]

# List of pair probes

# Rs bin probe results
L = {
    (0, 0.2) : 27.073,
    (0, 0.8) : 27.063,
    (1, 0.2) : 27.109,
    (1, 0.8) : 27.028,
    (2, 0.2) : 27.074,
    (2, 0.8) : 27.062,
    (3, 0.2) : 27.068,
    (3, 0.8) : 27.069,
    (4, 0.2) : 27.062,
    (4, 0.8) : 27.074,
    (5, 0.2) : 27.061,
    (5, 0.8) : 27.075,
    (6, 0.2) : 27.055,
    (6, 0.8) : 27.081,
    (7, 0.2) : 27.054,
    (7, 0.8) : 27.082,
    (8, 0.2) : 27.056,
    (8, 0.8) : 27.080,
    (9, 0.2) : 27.068,
    (9, 0.8) : 27.068,
}

for i,(k,v) in enumerate(L.items()):
    L[k] = v - 26.8650084

# Class weights
class_weights_dict = {
    99 : 2.002408,
    95 : 1.001044,
    92 : 1.001044,
    90 : 1.001044,
    88 : 1.001044,
    67 : 1.001044,
    65 : 1.001044,
    64 : 2.007104,
    62 : 1.001044,
    53 : 1.000000,
    52 : 1.001044,
    42 : 1.001044,
    16 : 1.001044,
    15 : 2.001886,
    6 : 1.001044,
}

# Build non-linear system of equations expression
system_str = ''

min_clip = 1e-15
max_clip = 1 - 1e-15


N_15 = 5663 + 328298 # From freq probe results
N_99 = 8209 + 318389 # From freq probe results

# Setup weight values
non_15_nor_99_weight_sum = 0
for k,v in class_weights_dict.items():
    if k!=15 and k!=99:
        non_15_nor_99_weight_sum += v
w_15 = 2.001886
w_99 = 2.002408

# Variable creation
N_99_x = [f'N_99_{b}' for b in range(10)]
N_15_x = [f'N_15_{b}' for b in range(10)]

for (bin_n, probe_p), probe_l in L.items():

    eq_str = ''

    # Add probe result (loss) term
    eq_str += f'{probe_p}  +  '

    # Build numerator
    num_str = ''

    # Sum weights except classes 90|99
    #num_str += f'{non_15_nor_99_weight_sum} * {np.log(min_clip)}  +  '

    # Build sum over class 15
    sum_of_N_15_x_except_N_15_b = ''
    for i, n15x in enumerate(N_15_x):
        if i!=bin_n:
            sum_of_N_15_x_except_N_15_b += n15x
        else:
            sum_of_N_15_x_except_N_15_b += '0'
        if i!=9:
            sum_of_N_15_x_except_N_15_b += ' + '

    num_str += f'{w_15} * ( {np.log(1-probe_p)} * {N_15_x[bin_n]} + '
    num_str += f'           {np.log(probe_p)  } * ({sum_of_N_15_x_except_N_15_b})'
    num_str += f'                                                                 ) / ({sum_of_N_15_x_except_N_15_b} + {N_15_x[bin_n]})  +  '

    # Build sum over class 99
    sum_of_N_99_x_except_N_99_b = ''
    for i, n99x in enumerate(N_99_x):
        if i != bin_n:
            sum_of_N_99_x_except_N_99_b += n99x
        else:
            sum_of_N_99_x_except_N_99_b += '0'
        if i != 9:
            sum_of_N_99_x_except_N_99_b += ' + '

    num_str += f'{w_99} * ( {np.log(probe_p)} * {N_99_x[bin_n]} + '
    num_str += f'           {np.log(1-probe_p)  } * ({sum_of_N_99_x_except_N_99_b})'
    num_str += f'                                                                 ) / ({sum_of_N_99_x_except_N_99_b} + {N_99_x[bin_n]})'

    # Wrap-up equation
    eq_str += f'('+num_str+f') / {non_15_nor_99_weight_sum + w_15 + w_99},'

    # Add to system
    system_str += eq_str

# Add class 15 and 99 total constraints
all_class_15_sum_str = ''
for var in N_15_x[:-1]:
    all_class_15_sum_str += var + ' + '
all_class_15_sum_str += N_15_x[-1]
system_str += f'{N_15:d} - (' + all_class_15_sum_str + '),'

all_class_99_sum_str = ''
for var in N_99_x[:-1]:
    all_class_99_sum_str += var + ' + '
all_class_99_sum_str += N_99_x[-1]
system_str += f'{N_99:d} - (' + all_class_99_sum_str + '),'

# Solve non-linear system of eqs

def eqs(variables):
    N_99_0,N_99_1,N_99_2,N_99_3,N_99_4,N_99_5,N_99_6,N_99_7,N_99_8,N_99_9,N_15_0,N_15_1,N_15_2,N_15_3,N_15_4,N_15_5,N_15_6,N_15_7,N_15_8,N_15_9 = variables

    return eval(system_str)

n_vars = 20

N_99_x.extend(N_15_x)
all_variables = N_99_x

# Naive initial guess building
x0 = np.random.rand(len(all_variables)) * 2000
upper_bd = np.ones(n_vars) * 150000
upper_bd[0] = 2000000
lower_bd = np.zeros(n_vars)
bds = (lower_bd, upper_bd)

res = least_squares(
    fun=eqs,
    x0=x0,
    #bounds=bds,
)

x_sol = res['x']
cost = res['cost']
residuals = res['fun']

# print('\nFrequencies :\n')
#
# for var_name, freq, resi in zip(all_variables, x_sol, residuals):
#     print(f'{var_name} : {freq/np.sum(x_sol):.3f} -- Residual : {resi:.4f}')

print('\nAbsolute numbers :\n')

for var_name, freq, resi in zip(all_variables, x_sol, residuals):
    print(f'{var_name} : {freq:.0f} -- Residual : {resi:.4f}')

print(f'\nTotal sum : {np.sum(x_sol)}')
print(f'\nCost :{cost}')

# # Print dict
# for var_name, freq, resi in zip(all_variables, x_sol, residuals):
#     var_name = '\''+var_name+'\''
#     print(f'{var_name} : {freq/np.sum(x_sol):.6f},')

# for eq in system_str.split(','):
#     print(eq)