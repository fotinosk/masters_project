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

    """
    for x_dot = 0:   ie no change to the state
        1. q = 0
        2. u = 0
        3. v = htta
        4. T = 3 * theta
    """

    if not danger:
        f_vel = (random.random()-0.5) * 0.2         # -0.2 - 0.2
        v_vel = (random.random()-0.5) * 0.4         # -0.2 - 0.2
        p_rate = (random.random()-0.5) * 0.2        # -0.1 - 0.1
        p_angle = (random.random()-0.5) * 0.3       # -0.3 - 0.3
        el_angle = 0
        thrust = 0

    else:
        f_vel = (random.random()-0.5) * 0.4         # -0.4 - 0.4
        v_vel = (random.random()-0.5)               # -0.5 - 0.5
        p_rate = (random.random()-0.5) * 0.3        # -0.2 - 0.2
        p_angle = (random.random()-0.5) * 0.5       # -0.5 - 0.5
        el_angle = 0
        thrust = 0

    return [f_vel, v_vel, p_rate, p_angle, el_angle, thrust]
