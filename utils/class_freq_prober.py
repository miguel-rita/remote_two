'''
Solve system of non-linear equations to determine class 99 frequencies after probing
'''

import numpy as np
from scipy.optimize import fsolve

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

# Pair galactic bin probe results
L = {
    (99, 90) : [28.931, 28.893, 28.898, 28.933],
    (95, 92) : [],
    (88, 67) : [],
    (65, 64) : [],
    (62, 53) : [],
    (52, 42) : [],
    (52, 42) : [],
    (16, 15) : [],
    (99, 6) : [],
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

# Individual and sum class weights
w99 = class_weights_dict[99]
w90 = class_weights_dict[90]
w_other = np.sum(list(class_weights_dict.values())) - w99 - w90


# Build non-linear system of equations expression
system_str = ''

min_clip = 1e-15
max_clip = 1 - 1e-15

for i, (l, p) in enumerate(zip(L, P)):
    eq_str = ''

    # Add probe result (loss) term
    eq_str += f'{l}  +  '

    # Build numerator
    num_str = ''

    # Sum weights except classes 90|99
    num_str += f'{w_other} * {np.log(min_clip)}  +  '

    # Build sum over class 90
    num_str += f'{w90} * (n0 * {np.log(1-p)} + n1 * {np.log(p)}) / (n0 + n1)  +  '

    # Build sum over class 99
    num_str += f'{w99} * (N0 * {np.log(p)} + N1 * {np.log(1-p)}) / (N0 + N1)     '

    # Wrap-up equation
    eq_str += f'('+num_str+f') / {w_other + w90 + w99},'

    # Add to system
    system_str += eq_str

# Add total constraint
system_str += '3492890 - t - n0-n1-N0-N1,'

print(system_str)

# Solve non-linear system of eqs
def eqs(vars):
    n0, n1, N0, N1, t = vars
    return (eval(system_str))

res = fsolve(eqs, np.ones(5) * 11000)


# # Print exact weights
# for _class, weight in zip(cd.keys(), res):
#     print(f'{_class} : {weight:.6f},')

for r in res:
    print(f'{r:.1f}')