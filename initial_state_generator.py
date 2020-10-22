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
        
    reasonable starting point: u = [1,3] x = [1,0,0,1]
    """

    if not danger:
        f_vel = 1 + (random.random()-0.5) * 0.2     # 0.9 - 1.1
        v_vel = (random.random()-0.5) * 0.4         # -0.2 - 0.2
        p_rate = (random.random()-0.5) * 0.2        # -0.1 - 0.1
        p_angle = 1 + (random.random()-0.5) * 0.3   # 0.7 - 1.3

    else:
        f_vel = 1 + (random.random()-0.5) * 0.4     # 0.6 - 1.4
        v_vel = (random.random()-0.5)               # -0.5 - 0.5
        p_rate = (random.random()-0.5) * 0.3        # -0.2 - 0.2
        p_angle = 1 + (random.random()-0.5) * 0.5   # 0.5 - 1.5

    return [f_vel, v_vel, p_rate, p_angle]
