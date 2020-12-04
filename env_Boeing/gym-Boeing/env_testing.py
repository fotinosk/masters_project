import gym
import gym_Boeing
import numpy as np

env = gym.make('normalized-danger-v0')

# for i in range(10):
#     print(env.flight.state)
#     env.reset()

a = env.action_space.sample()
print(type(a))
# print(a*100)

import torch

t = torch.Tensor([1,2])
print(t*100)
