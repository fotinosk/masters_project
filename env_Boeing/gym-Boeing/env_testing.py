import gym
import gym_Boeing
import numpy as np
import random

env = gym.make('boeing-danger-v2')


# a = env.action_space.sample()
s = env.reset(3)

# for _ in range(5002):
#     print(env.step(a))

# print(s)
print(env.flight.state)
