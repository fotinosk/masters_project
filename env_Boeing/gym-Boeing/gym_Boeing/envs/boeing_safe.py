import gym
from gym import error, spaces, utils
from gym.utils import seeding
from utils.flight import Flight
import sys
import numpy as np
import matplotlib.pyplot as plt


class BoeingSafe(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.done = False
        self.flight = Flight()
        self.observation = [0, 0, 0]
        self.past_sq_err = []

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-100, 100, shape=(2,), dtype=np.float32)

    def step(self, action):
        self.observation = self.flight.io(action)

        # sq_error = np.sum(np.square(self.observation))
        sq_error = np.linalg.norm(self.observation, 1)
        self.past_sq_err.append(sq_error)

        control_acc = 20  # hyperparameter
        control_len = 800  # hyperparameter

        # reward option 1
        # reward = 0
        # if sq_error < control_acc:
        #     reward += 10
        #     if len(self.past_sq_err) > control_len and self.past_sq_err[-control_len:] < [control_acc]:
        #         self.done = True
        #         reward = 100
        # else:
        #     reward -= sq_error

        # reward option 2
        # reward = 10 * np.exp(- 1e-6 * sq_error)

        # reward option 3
        # reward = 0
        # if len(self.past_sq_err) > control_len and max(self.past_sq_err[-control_len:]) < control_acc:
        #     print(f"Actions taken: {len(self.past_sq_err)}, Last Squared Error: {sq_error}")
        #     self.done = True
        #     reward = 100
        # # reward -= np.sqrt(sq_error)
        # reward -= sq_error

        # reward option 4 (l_1 norm)
        reward = 0
        if sq_error < control_acc:
            reward += 10
        if len(self.past_sq_err) > control_len and max(self.past_sq_err[-control_len:]) < control_acc:
            # print(f"Actions taken: {len(self.past_sq_err)}, Last Squared Error: {sq_error}")
            self.done = True
            reward = 100
        reward -= sq_error

        return self.observation, reward, self.done, {'len': len(self.past_sq_err), 'error': sq_error}

    def reset(self):
        self.flight.reset()
        self.observation = [0, 0, 0]
        self.past_sq_err = []
        self.done = False
        # print('Flight has been reset!')
        return self.observation

    def render(self, mode='human'):
        # self.flight.plot()
        x = list(np.arange(0, 0.05*len(self.past_sq_err), 0.05))
        plt.plot(x, self.past_sq_err)
        plt.xlabel('Time (sec)')
        plt.ylabel('Absolute Value of Deviations')
        plt.show(block=False)
        plt.pause(0.0001)

    def close(self):
        self.done = True
        self.reset()
        sys.exit()
