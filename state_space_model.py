"""
Module for user controlled input to the model.
User constantly inputs data using slider and output is plotted on graph.
"""

from control.matlab import *
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from actuator import Actuator

A = [[0, 1, 0, -3],
     [0, -3, 0.5, 0],
     [1, -10, -4, 0],
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

# sign = signal.lti(A, B, C, D)
sys = StateSpace(A, B, C, D)

input_time_delays = [0.1, 3.5]
dt = 0.05  # must be the same as that used in the user input or the RL model

actuator = Actuator(input_time_delays, dt)


def model(time, inputs, state, plot=False):
    """
     :param plot: If true then generate plots
     :param time: Array containing the time intervals
     :param inputs: Array containing the inputs
     :param state: Array containing the state
     :return: outputs and last state
     """
    inputs = actuator.io(inputs)  # delays inputs

    # t, yout, xout = sign.output(U=inputs, T=time, X0=state)
    yout, t, xout = lsim(sys, U=inputs, T=time, X0=state)

    if plot:
        y0 = [z[0] for z in yout]
        y1 = [z[1] for z in yout]
        y2 = [z[2] for z in yout]

        ax1 = plt.subplot(311)
        # plt.plot(time, y0)
        ax1.set_ylabel('Pitch Rate')
        plt.scatter(time, y0, s=2)

        ax2 = plt.subplot(312, sharex=ax1)
        # plt.plot(time, y1)
        ax2.set_ylabel('Pitch Angle')
        plt.scatter(time, y1, s=2)

        ax3 = plt.subplot(313, sharex=ax1)
        # plt.plot(time, y2)
        ax3.set_ylabel('Vertical Acceleration')
        plt.scatter(time, y2, s=2)

        plt.show(block=False)  # needed so that it the rest of the program can run
        plt.pause(0.05)

    return yout, xout[-1], time
