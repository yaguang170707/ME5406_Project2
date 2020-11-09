"""
A gym environment for stabilising a frictionless double-inverted pendulum system. The equations of motion were derived
by Yi, Yubazaki and Hirota 2001.
url: https://www.sciencedirect.com/science/article/abs/pii/S0954181001000218
"""

import gym
from gym import spaces
import numpy as np
from gym.utils import seeding

# import problem parameters
from problem_parameters import *


class doubleInvertedPendulum(gym.Env):
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
        Type: Discrete(3)
        Num     Action
        0       Applying a left-pointing force
        1       Do nothing
        2       Applying a right-pointing force

    Reward:
        If the two arms are kept upright (within the thresholds), then a reward of 1 will be given

    Starting State:
        All observations are initialised with a perturbation within [-0.05..0.05]

    Episode Termination:
        Robot reaches horizontal thresholds;
        The amplitude of the upper arm angle reaches beta_limit;
        Robot survives through 500 steps.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        490.0 over 100 consecutive trials.
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

        # bounds of state space
        high = np.array([x_limit,
                         np.finfo(np.float32).max,
                         alpha_limit,
                         np.finfo(np.float32).max,
                         beta_limit,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        # action space and state space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # create seed for random initialisation
        self.seed()

        # initialise viewer and state
        self.viewer = None
        self.state = None

    # create seed for random initialisation
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # simulate one step based on given action
    def step(self, action):
        # get state variables
        x, x_dot, alpha, alpha_dot, beta, beta_dot = self.state

        # calculate force
        F = (action-1)*force

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
            alpha = (alpha + alpha_dot*dt)#%(2*np.pi)
            beta = (beta+beta_dot*dt)#%(2*np.pi)
            x_dot += x_2dot*dt
            alpha_dot += alpha_2dot*dt
            beta_dot += beta_2dot*dt

        else:
            x_dot += x_2dot*dt
            alpha_dot += alpha_2dot*dt
            beta_dot += beta_2dot*dt
            x = x + x_dot*dt
            alpha = (alpha + alpha_dot*dt)#%(2*np.pi)
            beta = (beta + beta_dot*dt)#%(2*np.pi)

        self.state = np.array([x, x_dot, alpha, alpha_dot, beta, beta_dot])

        done = abs(x)>x_limit or abs(alpha)>alpha_limit or abs(beta)>beta_limit

        if not done:
            reward = 1
        else:
            reward = -100

        return self.state, reward, done, {}

    def reset(self):
        """Reset the simulation with small perturbations"""
        self.state = self.np_random.uniform(low=-0.02, high=0.02, size=(6,))
        return self.state

    def render(self, mode='human'):
        """ visualise the system"""

        # set window sizes
        w_window = 600
        h_window = 500

        # 'real' width and height of the robot
        robo_w = L1
        robo_h = 0.5*robo_w

        # set scaling factor
        w_world = 3 * x_limit + robo_w
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
        robo_y = d_wheel/2 + h_robo/2 + h_window/2

        # track vertical location
        track_y = robo_y - d_wheel/2 - h_robo/2

        # Initialise a Viewer object if not exist
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            # create viewer with pre-defined window sizes
            self.viewer = rendering.Viewer(w_window, h_window)

            # set sides values of the robot
            l, r, t, b = -w_robo/2, w_robo/2, h_robo/2, -h_robo/2
            # create a graphical robot
            self.robo = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # create transform attribute and add it to the robot
            self.robo_trans = rendering.Transform()
            self.robo.add_attr(self.robo_trans)
            # add robo in viewer
            self.viewer.add_geom(self.robo)

            # create a left wheel and add it in viewer
            self.wheel_l = rendering.make_circle(d_wheel/2)
            self.wheel_l_trans = rendering.Transform(translation=(-w_robo/4, -h_robo/2))
            self.wheel_l.add_attr(self.wheel_l_trans)
            self.wheel_l.add_attr(self.robo_trans)
            self.wheel_l.set_color(.5, .5, .5)
            self.viewer.add_geom(self.wheel_l)

            # create a right wheel and add it in viewer
            self.wheel_r = rendering.make_circle(d_wheel/2)
            self.wheel_r_trans = rendering.Transform(translation=(w_robo/4, -h_robo/2))
            self.wheel_r.add_attr(self.wheel_r_trans)
            self.wheel_r.add_attr(self.robo_trans)
            self.wheel_r.set_color(.5, .5, .5)
            self.viewer.add_geom(self.wheel_r)

            # create the track
            self.track = rendering.Line((0, track_y), (w_window, track_y))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            # set sides values of the lower arm
            l, r, t, b = -w_arm/2, w_arm/2, s_L1, 0

            # create a graphical lower arm
            self.arm_l = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.arm_l.set_color(.8, .6, .4)

            # create transform attribute and add it to the lower arm
            self.arm_l_trans = rendering.Transform()
            self.arm_l.add_attr(self.arm_l_trans)
            self.arm_l.add_attr(self.robo_trans)

            # add lower arm in viewer
            self.viewer.add_geom(self.arm_l)

            # set corner values of the upper arm
            l, r, t, b = -w_arm/2, w_arm/2, s_L2, 0

            # create a graphical upper arm
            self.arm_u = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.arm_u.set_color(.8, .6, .4)

            # create transform attribute and add it to the upper arm
            self.arm_u_trans = rendering.Transform(translation=(0, s_L1))
            self.arm_u.add_attr(self.arm_u_trans)
            self.arm_u.add_attr(self.arm_l_trans)
            self.arm_u.add_attr(self.robo_trans)

            # add upper arm in viewer
            self.viewer.add_geom(self.arm_u)

            # create a lower joint and add it in viewer
            self.joint_l = rendering.make_circle(w_arm/2)
            self.joint_l.add_attr(self.arm_l_trans)
            self.joint_l.add_attr(self.robo_trans)
            self.joint_l.set_color(.5, .5, .5)
            self.viewer.add_geom(self.joint_l)

            # create a upper joint and add it in viewer
            self.joint_u = rendering.make_circle(w_arm/2)
            self.joint_u.add_attr(self.arm_u_trans)
            self.joint_u.add_attr(self.arm_l_trans)
            self.joint_u.add_attr(self.robo_trans)
            self.joint_u.set_color(.5, .5, .5)
            self.viewer.add_geom(self.joint_u)

        if self.state is None:
            return None

        x, x_dot, alpha, alpha_dot, beta, beta_dot = self.state
        robo_x = x*scale + w_window / 2.0
        self.robo_trans.set_translation(robo_x, robo_y)
        self.arm_l_trans.set_rotation(alpha)
        self.arm_u_trans.set_rotation((beta-alpha))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None














