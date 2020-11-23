from wrappers import NormalizedActions
import gym
import gym_Boeing
import torch
from utils.replay_memory import ReplayMemory, Transition

env = gym.make('boeing-safe-v0')

from ddpg import DDPG

n = DDPG(0.99,0.01,[400,300], 2, env.action_space)

r = ReplayMemory(1000)

random_state = torch.Tensor([0.2,0.5,0.8])
random_action = torch.Tensor(env.action_space.sample())
done = torch.Tensor([False])
rand_next_state = torch.Tensor([1.2,1.5,-0.8])
reward = torch.Tensor([0.1])


for _ in range(int(1200)):
    r.push(random_state, random_action, done, rand_next_state, reward)

batch = r.sample(10)
batch = Transition(*zip(*batch))

n.update_params(batch)
