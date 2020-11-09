import numpy as np

# Amplitude of applied force
force = 0.5

# Gravitational acceleration
g = 9.8

# Mass of the cart
M = 0.5

# Mass of lower arm
m1 = 0.2

# Mass of upper arm
m2 = 0.2

# Length of lower arm
L1 = 0.6

# Length of upper arm
L2 = 0.6

# Gravity centre of lower arm
l1 = .5 * L1

# Gravity centre of upper arm
l2 = .5 * L2

# Horizontal movement boundary
x_limit = 1.5

# Lower arm degree limit
alpha_limit = 45*np.pi/180.

# upper arm degree limit
beta_limit = 20*np.pi/180.

# Time step
dt = 1/60.

# time-marching methods, euler or semi-implicit euler
marching_method = 'semi-euler'#'euler'#'semi-euler'

