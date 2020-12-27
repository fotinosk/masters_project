"""
Module that generates a random initial state for the plane.
"""

import random


def state(danger=False, ds = None):
    """
    :return: Returns a state array
    [forward velocity, vertical velocity, pitch rate, pitch angle].transpose
    If not in danger mode, then at least one of the states must be to the extremes.
    """

    f_vel = (random.random()-0.5) * 0.2         # -0.2 - 0.2
    v_vel = (random.random()-0.5) * 0.4         # -0.2 - 0.2
    p_rate = (random.random()-0.5) * 0.2        # -0.1 - 0.1
    p_angle = (random.random()-0.5) * 0.3       # -0.3 - 0.3
    el_angle = 0
    thrust = 0

    state = [f_vel, v_vel, p_rate, p_angle, el_angle, thrust]

    if danger:
        if ds is None:
            ds = random.randint(0,3)
        assert ds <= 3, 'Specified State Must be less than 3!'
        state[ds] += 10 
    return state
