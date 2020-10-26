import numpy as np

# Amplitude of applied force
force = 1

# Gravitational acceleration
g = 9.8

# Mass of the cart
M = 1

# Mass of lower arm
m1 = 1

# Mass of upper arm
m2 = 1

# Length of lower arm
L1 = 1

# Length of upper arm
L2 = 1

# Gravity centre of lower arm
l1 = .5 * L1

# Gravity centre of upper arm
l2 = .5 * L2

# Horizontal movement boundary
x_limit = 1.

# Lower arm degree limit
alpha_limit = 180*np.pi/180.

# upper arm degree limit
beta_limit = 180*np.pi/180.

# Time step
dt = 0.01

# time-marching methods, euler or semi-implicit euler
marching_method = 'euler'#'semi-euler'

