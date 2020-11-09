import time
import numpy as np

from problem_parameters import *


def import_test():
    """test the import parameters function is OK"""
    print(force, g, M, m1, m2, L1, L2, l1, l2, dt)


if __name__ == "__main__":
    # import_test()

    mode = 'human'

    screen_width = 600
    screen_height = 400

    world_width = 2.4 * 2
    scale = screen_width / (2*world_width)
    carty = 100  # TOP OF CART
    polewidth = 50
    polelen = 100
    cartwidth = 100
    cartheight = 100

    from gym.envs.classic_control import rendering

    viewer = rendering.Viewer(screen_width, screen_height)

    l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    axleoffset = cartheight / 4.0
    cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    carttrans = rendering.Transform()
    cart.add_attr(carttrans)
    viewer.add_geom(cart)

    l, r, t, b = -50,50,100-50,-50
    pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    pole.set_color(.8, .6, .4)
    poletrans = rendering.Transform(translation=(0, axleoffset))
    pole.add_attr(poletrans)
    # pole.add_attr(carttrans)
    viewer.add_geom(pole)

    carttrans.set_translation(200, 200)
    poletrans.set_translation(100, 100)
    # poletrans.set_rotation(0)
    # poletrans2.set_rotation(-np.pi*2/3)

    viewer.render(return_rgb_array=mode == 'rgb_array')

    time.sleep(10)




