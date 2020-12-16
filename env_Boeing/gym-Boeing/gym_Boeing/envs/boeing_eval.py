import gym
from gym import error, spaces, utils
from gym.utils import seeding
from utils.flight import Flight
import sys
import numpy as np
import matplotlib.pyplot as plt


class EvalDanger(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.done = False
        self.flight = Flight(danger=True)
        self.observation = [0, 0, 0]
        self.past_sq_err = []

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-100, 100, shape=(2,), dtype=np.float32)

    def step(self, action):
        self.observation = self.flight.io(action)

        sq_error = np.linalg.norm(self.observation, 1)
        self.past_sq_err.append(sq_error)

        control_acc = 20  # hyperparameter
        control_len = 800  # hyperparameter
        failure_time = 5000  #hyperparameter: max number of steps

        # reward option 4 (l_1 norm)
        reward = 0
        if sq_error < control_acc:
            reward += 10
        if len(self.past_sq_err) > control_len and max(self.past_sq_err[-control_len:]) < control_acc:
            self.done = True
            reward = 100
            print('Success!')
        elif len(self.past_sq_err) == failure_time:
            self.done = True
            reward = - 1000
            print('Failure!')
        reward -= sq_error

        return self.observation, reward, self.done, {'len': len(self.past_sq_err), 'error': sq_error}

    def reset(self, ds=None):
        self.flight.reset(ds)
        self.observation = [0, 0, 0]
        self.past_sq_err = []
        self.done = False
        return self.observation

    def render(self, mode='human'):
        # self.flight.plot()
        x = list(np.arange(0, 0.05*len(self.past_sq_err), 0.05))
        try:
            plt.plot(x, self.past_sq_err)
            plt.xlabel('Time (sec)')
            plt.ylabel('Absolute Value of Deviations')
            plt.show(block=False)
            plt.pause(0.0001)
        except Exception as e:
            print(f"Run into known Matplotlib bug, can't show plot. \n Error {e}")

    def close(self):
        self.done = True
        self.reset()
        sys.exit()
