"""
A gym environment for stabilising a frictionless double-inverted pendulum system. The equations of motion were derived
by Yi, Yubazaki and Hirota 2001.
url: https://www.sciencedirect.com/science/article/abs/pii/S0954181001000218
"""

import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
from gym.envs.classic_control import rendering

# import problem parameters
from problem_parameters import *


class DoubleInvertedPendulum(gym.Env):
    """
    Description:
        This is a custom environment, which defines a double inverted pendulum problem. There is no friction
        between the robot and the horizontal track, as well as the two arm joints.

    Source:
        This system is equivalent to the one described in Yi, Yubazaki and Hirota 2001.

    Observation:
        Type: Box(6)
        Num     Observation                                 Min             Max
        0       Robot horizontal position                   -x_limit        x_limit
        1       Robot horizontal velocity                   -inf            inf
        2       Angle between lower arm and vertical axis   -alpha_limit    alpha_limit
        3       Angular velocity of lower arm               -inf            inf
        4       Angle between upper arm and vertical axis   -beta_limit     beta_limit
        6       Angular velocity of upper arm               -inf            inf

        The min and max values, as well as other physical properties can be specified by user inputs in
        problem_parameters.py file.

    Action:
        Type: Discrete(N_a)

    Reward:
        A reward will be given for each step. The amount of the reward can be selected by comment/uncomment the desired
        one.

    Starting State:
        All observations are initialised with a perturbation within [-0.05..0.05]

    Episode Termination:
        Robot reaches horizontal thresholds;
        The amplitude of the upper arm angle reaches beta_limit;
        The amplitude of the lower arm angle reaches alpha_limit;
    """

    def __init__(self):
        # define coefficients of the dynamics equations as in (Yi, Yubazaki and Hirota 2001)

        # linear coefficients
        self.a11 = M + m1 + m2
        self.a22 = 4*m1*(l1**2)/3 + m2*(L1**2)
        self.a33 = 4*m2*(l2**2)/3

        # nonlinear coefficients
        self.a12_without_cos_a = m1*l1 + m2*L1
        self.a21_without_cos_a = self.a12_without_cos_a
        self.a13_without_cos_b = m2*l2
        self.a31_without_cos_b = self.a13_without_cos_b
        self.a23_without_cos_a_m_b = m2*L1*l2
        self.a32_without_cos_a_m_b = self.a23_without_cos_a_m_b

        # RHS coefficients
        self.b11 = m1*l1 + m2*L1
        self.b12 = m2*l2
        self.b21 = self.b11*g
        self.b22 = -m2*L1*l2
        self.b31 = m2*l2*g
        self.b32 = m2*L1*l2

        # maximum potential energy of the system
        self.P_max = m1*g*l1 + m2*g*(L1+l2)

        # bounds of state space
        high = np.array([x_limit,
                         np.finfo(np.float32).max,
                         alpha_limit,
                         np.finfo(np.float32).max,
                         beta_limit,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        # action space and state space
        self.action_space = spaces.Discrete(N_a)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # create seed for random initialisation
        self.seed()

        # initialise viewer and state
        self.viewer = None
        self.robot = None
        self.robot_ghost_l = None
        self.robot_ghost_r = None
        self.track = None
        self.state = None

    # create seed for random initialisation
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset the simulation with small perturbations"""
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        return self.state

    # calculate the kinetic energy of the system
    def T(self):
        x, x_dot, alpha, alpha_dot, beta, beta_dot = self.state
        T0 = 0.5 * M * x_dot ** 2
        T1 = 0.5 * m1 * x_dot ** 2 + 0.5 * (m1 * l1 ** 2 + I1) * alpha_dot ** 2 + m1 * l1 * alpha_dot * x_dot * np.cos(
            alpha)
        T2 = 0.5 * m2 * x_dot ** 2 + 0.5 * m2 * (L1 * alpha_dot) ** 2 + 0.5 * (m2 * l2 ** 2 + I2) * beta_dot ** 2 \
             + m2 * L1 * x_dot * alpha_dot * np.cos(alpha) + m2 * l2 * x_dot * beta_dot * np.cos(beta) \
             + m2 * L2 * l2 * alpha_dot * beta_dot * np.cos(alpha - beta)
        return T0 + T1 + T2

    def P(self):
        x, x_dot, alpha, alpha_dot, beta, beta_dot = self.state
        P1 = m1*g*l1*np.cos(alpha)
        P2 = m2*g*(L1*np.cos(alpha) + l2*np.cos(beta))
        return P1+P2

    # simulate one step based on given action
    def step(self, action):
        # get state variables
        x, x_dot, alpha, alpha_dot, beta, beta_dot = self.state

        # calculate force
        F = 2*F_max/(self.action_space.n-1) * action - F_max

        # update equation coefficients
        a12 = self.a12_without_cos_a * np.cos(alpha)
        a21 = a12
        a13 = self.a13_without_cos_b * np.cos(beta)
        a31 = a13
        a23 = self.a23_without_cos_a_m_b * np.cos(alpha - beta)
        a32 = a23

        b1 = F + self.b11*(alpha_dot**2)*np.sin(alpha) + self.b12*(beta_dot**2)*np.sin(beta)
        b2 = self.b21*np.sin(alpha) + self.b22*(beta_dot**2)*np.sin(alpha-beta)
        b3 = self.b31*np.sin(beta) + self.b32*(alpha_dot**2)*np.sin(alpha-beta)

        # solving linearised dynamics equations
        A = np.array([[self.a11, a12, a13], [a21, self.a22, a23], [a31, a32, self.a33]])
        b = np.array([b1, b2, b3])
        x_2dot, alpha_2dot, beta_2dot = np.linalg.solve(A, b)

        # update state variables using specified time marching method
        if marching_method == 'euler':
            x = x + x_dot*dt
            alpha = (alpha + alpha_dot*dt)
            beta = (beta+beta_dot*dt)
            x_dot += x_2dot*dt
            alpha_dot += alpha_2dot*dt
            beta_dot += beta_2dot*dt

        else:
            x_dot += x_2dot*dt
            alpha_dot += alpha_2dot*dt
            beta_dot += beta_2dot*dt
            x = x + x_dot*dt
            alpha = (alpha + alpha_dot*dt)
            beta = (beta + beta_dot*dt)

        # reward schemes, select one you like
        reward = 1
        # reward = self.P() / self.P_max
        # reward = (self.P() - self.T()) / self.P_max
        # reward = np.exp((self.P() - self.T()) / self.P_max - 1)

        self.state = np.array([x, x_dot, alpha, alpha_dot, beta, beta_dot])
        observations = self.state.copy()
        done = abs(alpha) > alpha_limit or abs(beta) > beta_limit or abs(x) > x_limit

        # enable boundary ignore
        if ignore_x_limit:
            observations[0] = 0.
            done = abs(alpha)>alpha_limit or abs(beta)>beta_limit

        return observations, reward, done, {}

    def render(self, mode='human'):
        """ visualise the system"""

        # set window sizes
        w_window = 600
        h_window = 500

        # 'real' width and height of the robot
        robo_w = L1
        robo_h = 0.5*robo_w

        # set scaling factor
        w_world = 2 * x_limit
        scale = w_window/w_world

        # width of arm in the window
        w_arm = scale*L1/10

        # length of lower arm in the window
        s_L1 = scale*L1

        # length of upper arm in the window
        s_L2 = scale*L2

        # width of robot in the window
        w_robo = scale*robo_w

        # height of robot in the window
        h_robo = scale*robo_h

        # diameter of wheels in the window
        d_wheel = w_robo/4.

        # robo vertical location
        robo_y = d_wheel/2 + h_robo/2 + h_window/5

        # track vertical location
        track_y = robo_y - d_wheel/2 - h_robo/2

        # Initialise a Viewer object if not exist
        if self.viewer is None:

            # create viewer with pre-defined window sizes
            self.viewer = rendering.Viewer(w_window, h_window)

            # create robot object and ghost robots for periodic display
            self.robot = Robot(w_robo, h_robo, d_wheel, w_arm, s_L1, s_L2)
            self.robot.add_to(self)

            # plot ghost robots only if x_limit is ignored
            if ignore_x_limit:
                self.robot_ghost_l = Robot(w_robo, h_robo, d_wheel, w_arm, s_L1, s_L2)
                self.robot_ghost_l.add_to(self)

                self.robot_ghost_r = Robot(w_robo, h_robo, d_wheel, w_arm, s_L1, s_L2)
                self.robot_ghost_r.add_to(self)

            # create the track
            self.track = rendering.Line((0, track_y), (w_window, track_y))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x, x_dot, alpha, alpha_dot, beta, beta_dot = self.state
        if ignore_x_limit:
            x = np.mod((x + x_limit), 2 * x_limit) - x_limit
        robo_x = x*scale + w_window / 2.0
        self.robot.cart_trans.set_translation(robo_x, robo_y)
        self.robot.arm_l_trans.set_rotation(alpha)
        self.robot.arm_u_trans.set_rotation((beta-alpha))

        # plot ghost robots only if x_limit is ignored
        if ignore_x_limit:
            self.robot_ghost_l.cart_trans.set_translation(robo_x-w_window, robo_y)
            self.robot_ghost_l.arm_l_trans.set_rotation(alpha)
            self.robot_ghost_l.arm_u_trans.set_rotation((beta-alpha))

            self.robot_ghost_r.cart_trans.set_translation(robo_x+w_window, robo_y)
            self.robot_ghost_r.arm_l_trans.set_rotation(alpha)
            self.robot_ghost_r.arm_u_trans.set_rotation((beta-alpha))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # close the environment
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# define a class that handles the creation of graphical objects for visualisation
class Robot:
    def __init__(self, w_robo, h_robo, d_wheel, w_arm, s_L1, s_L2):
        self.cart, self.cart_trans = self.create_cart(w_robo, h_robo)
        self.wheel_l, self.wheel_l_trans = self.create_wheel(w_robo, h_robo, d_wheel, -1)
        self.wheel_r, self.wheel_r_trans = self.create_wheel(w_robo, h_robo, d_wheel, 1)
        self.arm_l, self.arm_l_trans = self.create_arm_l(w_arm, s_L1)
        self.arm_u, self.arm_u_trans = self.create_arm_u(w_arm, s_L1, s_L2)
        self.joint_l = self.create_joint_l(w_arm)
        self.joint_u = self.create_joint_u(w_arm)

    # create the cart
    def create_cart(self, w_robo, h_robo):
        # set sides values of the robot
        l, r, t, b = -w_robo / 2, w_robo / 2, h_robo / 2, -h_robo / 2
        # create a graphical robot
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        # create transform attribute and add it to the robot
        cart_trans = rendering.Transform()
        cart.add_attr(cart_trans)
        return cart, cart_trans

    # create wheels of the cart
    def create_wheel(self, w_robo, h_robo, d_wheel, offset):
        wheel = rendering.make_circle(d_wheel / 2)
        wheel_trans = rendering.Transform(translation=(offset*w_robo / 4, -h_robo / 2))
        wheel.add_attr(wheel_trans)
        wheel.add_attr(self.cart_trans)
        wheel.set_color(.5, .5, .5)
        return wheel, wheel_trans

    # create lower arm
    def create_arm_l(self, w_arm, s_L1):
        # set sides values of the lower arm
        l, r, t, b = -w_arm / 2, w_arm / 2, s_L1, 0

        # create a graphical lower arm
        arm_l = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        arm_l.set_color(.8, .6, .4)

        # create transform attribute and add it to the lower arm
        arm_l_trans = rendering.Transform()
        arm_l.add_attr(arm_l_trans)
        arm_l.add_attr(self.cart_trans)

        return arm_l, arm_l_trans

    # create upper arm
    def create_arm_u(self, w_arm, s_L1, s_L2):
        # set corner values of the upper arm
        l, r, t, b = -w_arm / 2, w_arm / 2, s_L2, 0

        # create a graphical upper arm
        arm_u = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        arm_u.set_color(.8, .6, .4)

        # create transform attribute and add it to the upper arm
        arm_u_trans = rendering.Transform(translation=(0, s_L1))
        arm_u.add_attr(arm_u_trans)
        arm_u.add_attr(self.arm_l_trans)
        arm_u.add_attr(self.cart_trans)

        return arm_u, arm_u_trans

    # create joints
    def create_joint_l(self, w_arm):
        # create a lower joint and add it in viewer
        joint_l = rendering.make_circle(w_arm / 2)
        joint_l.add_attr(self.arm_l_trans)
        joint_l.add_attr(self.cart_trans)
        joint_l.set_color(.5, .5, .5)
        return joint_l

    def create_joint_u(self, w_arm):
        # create a upper joint and add it in viewer
        joint_u = rendering.make_circle(w_arm / 2)
        joint_u.add_attr(self.arm_u_trans)
        joint_u.add_attr(self.arm_l_trans)
        joint_u.add_attr(self.cart_trans)
        joint_u.set_color(.5, .5, .5)
        return joint_u

    # add the objects to viewer of the environment
    def add_to(self, env):
        env.viewer.add_geom(self.cart)
        env.viewer.add_geom(self.wheel_l)
        env.viewer.add_geom(self.wheel_r)
        env.viewer.add_geom(self.arm_l)
        env.viewer.add_geom(self.arm_u)
        env.viewer.add_geom(self.joint_l)
        env.viewer.add_geom(self.joint_u)













