import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt

class SimpleModel(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.done = False
        self.past_err = []
        self.b = random.choice([-1,1])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,))
        self.action_space = spaces.Box(-1,1, shape=(1,))
        self.state = random.random() * 10 - 5

    def reset(self):
        print('Resetting...')
        self.done = False
        self.past_err = []
        self.b = random.choice([-1,1])
        self.state = random.random() * 10
        return self.state

    def step(self, action):
        self.state = self.state + self.b * action
        error = abs(self.state)
        self.past_err.append(error)

        control_acc = 2
        control_len = 50
        failure_len = 500

        reward = 0 

        if error < control_acc:
            reward += 10 
        if len(self.past_err) > control_len and max(self.past_err[-control_len:]) < control_acc:
            self.done = True
            reward += 100
            print('SUCCESS!')
        elif len(self.past_err) >= failure_len:
            self.done = True
            reward -= 100
            print('FAILURE.')
        reward -= error

        return self.state, reward, self.done, {'len': len(self.past_err), 'error': error}

    def close(self):
        self.done = True
        self.reset()

