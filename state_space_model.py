"""
Module for user controlled input to the model.
User constantly inputs data using slider and output is plotted on graph.
"""

import control
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

A = [[0.00187, 0.0263, -86.15, -31.939],
     [-0.07, -0.2941, 672.9, -4.12],
     [0.002653, -0.0009834, -0.3449, 0.0003729],
     [0, 0, 1, 0]]
B = [[1.93],
     [-15.21],
     [-0.9686],
     [0]]
C = [[0, 0, 0, 1],
     [0, 0, 1, 0],
     [-0.07, -0.2941, 672.9, -4.12]]
D = [[0],
     [0],
     [-15.21]]

sign = signal.lti(A, B, C, D)


# print(sign.output([10, 10, 10, 30, 300, 10, 20, 50], [1, 2, 3, 4, 5, 6, 7, 8]))
# inputs and timestamps and outputs the output and state


def model(time, inputs, state):
    """
     :param time: Array containing the time intervals
     :param inputs: Array containing the inputs
     :param state: Array containing the state
     :return: outputs and last state, plot(?)
     """
    t, yout, xout = sign.output(inputs, time, X0=state)

    y0 = [z[0] for z in yout]
    y1 = [z[1] for z in yout]
    y2 = [z[2] for z in yout]

    plot0 = plt.figure(1)
    plt.plot(time, y0)
    plot1 = plt.figure(2)
    plt.plot(time, y1)
    plot2 = plt.figure(3)
    plt.plot(time, y2)
    plt.show(block=False)  # needed so that it the rest of the program can run
    plt.pause(0.05)

    return yout, xout[-1], time
