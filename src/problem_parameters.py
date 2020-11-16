# This is the input file for setting up the physical parameters of the problem
import numpy as np

# number of force levels
N_a = 11

# Amplitude of applied force
F_max = 100.

# Gravitational acceleration
g = 9.8

# Mass of the cart
M = 1.5

# Mass of lower arm
m1 = 0.5

# Mass of upper arm
m2 = 0.5

# Length of lower arm
L1 = 0.5

# Length of upper arm
L2 = 0.5

# Gravity centre of lower arm
l1 = .5 * L1

# Gravity centre of upper arm
l2 = .5 * L2

# moment of inertial of the lower arm
I1 = m1*L1/12.

# moment of inertial of the upper arm
I2 = m2*L2/12.

# Horizontal movement boundary and the switch for ignoring it
x_limit = 1.5*(L1+L2)
ignore_x_limit = False  # if true, the x_limit values is only used for visualisation purposes

# Lower arm degree limit
alpha_limit = 20*np.pi/180.

# upper arm degree limit
beta_limit = 15*np.pi/180.

# Time step
dt = 0.02

# time-marching methods, euler or semi-implicit euler
marching_method = 'semi-euler'  # 'euler'

