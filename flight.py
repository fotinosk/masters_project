"""Flight"""

from initial_state_generator import state
from control.matlab import *
import matplotlib.pyplot as plt
from model_parameters import A, B, C, D, dt
import numpy as np

# TODO: Implement both inputs (actions) and outputs

class Flight:
    """
    Flight class that will be used by RL model
    Initialized with a random initial state
    """

    def __init__(self, f_dt=dt, danger=False):
        self.dt = f_dt
        self.danger = danger
        self.state = state(self.danger)
        self.t = 0
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.sys = StateSpace(self.A, self.B, self.C, self.D)
        self.yout = None
        self.last_input = [0, 0]
        self.elevator_range = np.pi
        self.thrust_range = 10
        self.track_outs = []

    def reset(self):
        """Reset the flight"""

        self.t = 0
        self.state = state(self.danger)
        self.yout = None
        self.state = state(self.danger)
        self.last_input = [0, 0]
        self.track_outs = []

    def plot(self):
        """Plot the results"""

        timeline = [i * self.dt for i in range(len(self.track_outs))]

        y0 = [z[0] for z in self.track_outs]
        y1 = [z[1] for z in self.track_outs]
        y2 = [z[2] for z in self.track_outs]

        ax1 = plt.subplot(311)
        ax1.set_ylabel('Pitch Rate')
        plt.scatter(timeline, y0, s=2)

        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_ylabel('Pitch Angle')
        plt.scatter(timeline, y1, s=2)

        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_ylabel('Vertical Acceleration')
        plt.scatter(timeline, y2, s=2)

        plt.show()

    def io(self, inputs):
        """Input Output for the system"""

        yout, _, xout = lsim(self.sys, U=[self.last_input, inputs], T=[self.t, self.t + self.dt], X0=self.state)
        self.last_input = inputs[-1]
        self.t += self.dt
        self.state = xout[-1]
        self.yout = yout[-1]
        self.last_input = inputs
        self.track_outs.append(self.yout)
        return self.yout

    def get_actions(self):
        """Returns arrays of actions for each of the inputs"""

        incr = 0.1
        thrust_action_space = np.arange(-self.thrust_range, self.thrust_range, incr)
        elevator_action_space = np.arange(-self.elevator_range, self.elevator_range, incr)

        return elevator_action_space, thrust_action_space

    def show_outputs(self):
        """Show chronological outputs"""
        print(self.track_outs)






