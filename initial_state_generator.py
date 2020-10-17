"""
Module that generates a random initial state for the plane.
"""

import random


def state(danger=False):
    """
    :return: Returns a state array
    [forward velocity, vertical velocity, pitch rate, pitch angle].transpose
    If not in danger mode, then at least one of the states must be to the extremes.
    """

    if not danger:
        f_vel = random.randint(260, 320)  # typical cruising speed of 293m/s
        v_vel = random.randint(-12, 12)  # typical descent speed of 9m/s
        p_rate = 10 * random.random() - 5  # guessed pitch rates between [-5, 5]
        p_angle = random.randint(-10, 10)  # typical pitch angle of 0 deg

    else:
        f_vel = random.randint(50, 350)
        v_vel = random.randint(-50, 20)
        p_rate = 15 * random.random() - 10  # [-10, 5]
        p_rate = round(p_rate, 3)
        p_angle = random.randint(-20, 20)

    return [[f_vel], [v_vel], [p_rate], [p_angle]]
