from wrappers import NormalizedActions
import gym
import gym_Boeing
import torch

env = gym.make('boeing-safe-v0')
n = NormalizedActions(env)

x = torch.randn(2)

un = 100 * x
nor = n.action(un)
un2 = n.reverse_action(nor)

print(x,un, nor, un2 )

