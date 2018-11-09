'''
Solve system of non-linear equations to determine class 99 frequencies after probing
'''

import numpy as np
from scipy.optimize import least_squares

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
    (16, 15),
    (99, 6),
]

# Pair galactic bin probe results
L = {
    (99, 90) : [28.931, 28.893, 28.898, 28.933],
    (95, 92) : [30.713, 30.741, 30.779, 30.835],
    (88, 67) : [30.835, 30.788, 30.778, 30.778],
    (65, 64) : [29.158, 28.974, 28.889, 28.833],
    (62, 53) : [30.715, 30.743, 30.780, 30.837],
    (52, 42) : [30.835, 30.788, 30.779, 30.788],
    (16, 15) : [29.167, 28.984, 28.899, 28.843],
    (99, 6)  : [28.809, 28.845, 28.898, 28.980],
}

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

# Individual and sum class weights for each class pair
w_pairs = [
    (
        class_weights_dict[p[0]],
        class_weights_dict[p[1]],
        np.sum(list(class_weights_dict.values())) - class_weights_dict[p[0]] - class_weights_dict[p[1]],
    )

    for p in pairs
]

# Build non-linear system of equations expression
system_str = ''

min_clip = 1e-15
max_clip = 1 - 1e-15

all_variables = []

for (class_a, class_b), (w_a, w_b, w_other) in zip(pairs, w_pairs):

    # Variable creation
    g_a = f'g_{class_a}'
    g_b = f'g_{class_b}'
    eg_a = 'e' + g_a
    eg_b = 'e' + g_b

    if class_a == 99 and class_b == 6: # odd number of classes thus class 99 appears twice, must not add it again
        all_variables.extend([g_b, eg_b])
    else:
        all_variables.extend([g_a, g_b, eg_a, eg_b])

    for i, p in enumerate(P):
        eq_str = ''

        # Add probe result (loss) term
        eq_str += f'{L[(class_a, class_b)][i]}  +  '

        # Build numerator
        num_str = ''

        # Sum weights except classes 90|99
        num_str += f'{w_other} * {np.log(min_clip)}  +  '

        # Build sum over class a
        num_str += f'{w_a} * ({g_a} * {np.log(p)} + {eg_a} * {np.log(1-p)}) / ({g_a} + {eg_a})  +  '

        # Build sum over class b
        num_str += f'{w_b} * ({g_b} * {np.log(1-p)} + {eg_b} * {np.log(p)}) / ({g_b} + {eg_b})     '

        # Wrap-up equation
        eq_str += f'('+num_str+f') / {w_other + w_a + w_b},'

        # Add to system
        system_str += eq_str

# Add total constraint
all_var_sum_str = ''

for var in all_variables[:-1]:
    all_var_sum_str += var + ' + '
all_var_sum_str += all_variables[-1]

num_samples = 3492890
system_str += f'{num_samples:d} - (' + all_var_sum_str + '),'

# Add number of galactic sources constraint
gal_var_sum_str = ''

gal_vars = [v for v in all_variables if 'eg' not in v]
for gal_var in gal_vars[:-1]:
    gal_var_sum_str += gal_var + ' + '
gal_var_sum_str += gal_vars[-1]

num_galactic_samples = 390510
system_str += f'{num_galactic_samples:d} - (' + gal_var_sum_str + '),'

# Solve non-linear system of eqs

# all_var_list_str = ''
#
# for var in all_variables[:-1]:
#     all_var_list_str += var + ', '
# all_var_list_str += all_variables[-1]

def eqs(variables):
    g_99,g_90,eg_99,eg_90,g_95,g_92,eg_95,eg_92,g_88,g_67,eg_88,eg_67,g_65,g_64,eg_65,eg_64,g_62,g_53,eg_62,eg_53,g_52,g_42,eg_52,eg_42,g_16,g_15,eg_16,eg_15,g_6,eg_6 = variables
    return eval(system_str)

n_vars = len(all_variables)

# Naive initial guess building
gal_xo_per_class = num_galactic_samples / 7
egal_xo_per_class = (num_samples - num_galactic_samples) / 30
x0 = np.zeros(n_vars)
x0[::4] = gal_xo_per_class
x0[1::4] = gal_xo_per_class
x0[2::4] = egal_xo_per_class
x0[3::4] = egal_xo_per_class
x0 = np.ones(n_vars) * 100000
res = least_squares(
    fun=eqs,
    x0=x0,
    bounds=(np.zeros(n_vars), np.ones(n_vars) * num_samples/9),
)

x_sol = res['x']
cost = res['cost']
residuals = res['fun']

print('\nFrequencies :\n')

for var_name, freq, resi in zip(all_variables, x_sol, residuals):
    print(f'{var_name} : {freq/np.sum(x_sol):.3f} -- Residual : {resi:.4f}')

print('\nAbsolute numbers :\n')

for var_name, freq, resi in zip(all_variables, x_sol, residuals):
    print(f'{var_name} : {freq:.0f} -- Residual : {resi:.4f}')

print(f'\nTotal sum : {np.sum(x_sol)}')
print(f'\nCost :{cost}')

# # Print dict
# for var_name, freq, resi in zip(all_variables, x_sol, residuals):
#     var_name = '\''+var_name+'\''
#     print(f'{var_name} : {freq/np.sum(x_sol):.6f},')