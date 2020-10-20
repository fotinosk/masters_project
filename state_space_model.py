"""
Module for user controlled input to the model.
User constantly inputs data using slider and output is plotted on graph.
"""

import control
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

A = [[0, 1, 0, -3],
     [0, -3, 0.5, 0],
     [1, -10, 4, 0],
     [0, 0, 1, 0]]

B = [[0, 1],
     [0, 0],
     [-1, 0],
     [0, 0]]

C = [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, -3, 0.5, 0]]

D = [[0, 0],
     [0, 0],
     [0, 0]]

sign = signal.lti(A, B, C, D)


# print(sign.output([10, 10, 10, 30, 300, 10, 20, 50], [1, 2, 3, 4, 5, 6, 7, 8]))
# print(sign.output([[1, 1], [1, 1], [1, 1]], [1, 2, 3]))
# inputs and timestamps and outputs the output and state


def model(time, inputs, state):
    """
     :param time: Array containing the time intervals
     :param inputs: Array containing the inputs
     :param state: Array containing the state
     :return: outputs and last state, plot(?)
     """
    t, yout, xout = sign.output(inputs, time, X0=state)

    # y0 = [z[0] for z in yout]
    # y1 = [z[1] for z in yout]
    # y2 = [z[2] for z in yout]
    #
    # plot0 = plt.figure(1)
    # plt.plot(time, y0)
    # plot1 = plt.figure(2)
    # plt.plot(time, y1)
    # plot2 = plt.figure(3)
    # plt.plot(time, y2)

    plt.show(block=False)  # needed so that it the rest of the program can run
    plt.pause(0.05)

    return yout, xout[-1], time
