"""Convolve unit pulse with exponential and return truncated result

Assume the pulse lasts 0.05sec, ie the sampling time of the slider
"""

import numpy as np

dt = 0.05


def exp_convolved_with_step(dt, time_lag, pulse_height):
    """Returns the truncated waveform of the convolved function"""
    time_lag = 1 / time_lag

    def trunc(t):
        """Returns the value of the convoluted function at that point in time"""
        if t < dt:
            return 1 - np.exp(-time_lag * t)
        else:
            return (1 - np.exp(-time_lag * dt)) * np.exp(-time_lag * (t - dt))

    # Now we have the function that represents the convolution between the exponential function (ie lag) and
    # the unit pulse
    # TODO: Discretize and distribute across the future inputs, by sliding scale and adding


print(exp_convolved_with_step(0.05, 0.1, 1))