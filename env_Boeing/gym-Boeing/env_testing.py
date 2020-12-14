import gym
import gym_Boeing
import numpy as np

env = gym.make('boeing-danger-v2')


a = env.action_space.sample()
s = env.reset()

for _ in range(5002):
    print(env.step(a))
