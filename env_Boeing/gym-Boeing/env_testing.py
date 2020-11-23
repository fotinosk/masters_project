import gym
import gym_Boeing
import numpy as np

env = gym.make('boeing-danger-v0')

for i in range(10):
    print(env.flight.state)
    env.reset()
