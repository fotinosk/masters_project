"""
This module contains the 3rd failure mode of the plane. 
The B matrix is now 0.5 its original value, as well as a scenario with inverted controls.
The enviroment will also randomly give impulese. 
"""

from control.matlab import *
from utils.model_parameters import A, B, C, D, dt
from utils.prints import print_green, print_red
import numpy as np
from utils.flight_v2 import Flight
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import numpy as np
import matplotlib.pyplot as plt


B_err = 0.5 * B
neg_B = -1 * B


class FailureMode5(gym.Env):
    """
    Gym enviroment containing the normal modeo of operation and on with an ineffective B matrix (0.8B)
    """
    

    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Initialize the enviroment
        """        
        self.done = False
        self.flight = Flight(failure_modes=[[A,B_err,C,D],[A,neg_B,C,D]])
        self.possibilities = self.flight.possibilities
        self.observation = [0,0]
        self.past_err = []

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))
        self.actual_space = np.array([10,10]) # rescale actions to avoid needing to normalize the enviroment

    def step(self, action):
        """
        Handles the io of the enviroment

        Args:
            action (list, ndarray, tensor): The action taken be the control model

        Returns:
            tuple: Returns the state following the action, the reward, whether the episode is finished and an information dict.
        """  
        action *= self.actual_space
        self.observation = self.flight.io(action)
        error = np.linalg.norm(self.observation, 1)
        self.past_err.append(error)

        control_acc  = 20
        control_len  = 800
        failure_time = 5000

        reward = 0 

        if error < control_acc:
            reward += 10
        
        if len(self.past_err) > control_len and max(self.past_err[-control_len:]) < control_acc:
            self.done = True
            reward = 100
            print_green(f"Episode Succesful| Episode Length: {len(self.past_err)}")
        elif len(self.past_err) == failure_time:
            self.done = True
            reward = -1000
            print_red('Episode Failed')

        reward -= error

        return self.observation, reward, self.done, {'len': len(self.past_err), 'error': error, 'action': action}

    def reset(self, ds=None):
        """
        Resets the enviroment

        Args:
            ds (int, optional): Whether any particular initial state of the enviroment is needed (used for RL evaluation). Defaults to None.

        Returns:
            observation list: the initial state of the enviroment
        """        
        self.flight.reset(ds)
        self.observation = [0,0,0]
        self.past_err = []
        self.done = False
        return self.observation

    def render(self, mode='human', block=False, stack=False):
        """
        Produces plots for the enviroment.
        """        
        x = list(np.arange(0, 0.05*len(self.past_err), 0.05))
        try:
            if not stack:
                plt.cla()
            plt.plot(x, self.past_err)
            plt.xlabel('Time (sec)')
            plt.ylabel('Absolute Value of Deviations')
            plt.show(block=block)
            plt.pause(0.01)
        except Exception:
            print("Run into known Matplotlib bug, can't show plot.")

    def close(self):
        """
        Close enviroment
        """        
        self.done = True
        self.reset()
        sys.exit()
