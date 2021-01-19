import argparse
import logging
import os
from tqdm import tqdm

import gym
import gym_Boeing
import numpy as np
import torch

from ddpg import DDPG
from wrappers import NormalizedActions
from augment import Augment

# Parse given arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="boeing-danger-v1",
                    help="Env. on which the agent should be trained")
parser.add_argument("--render", default="True", help="Render the steps")
parser.add_argument("--seed", default=0, help="Random seed")
parser.add_argument("--save_dir", default="./saved_models_augmented_states/", help="Dir. path to load a model")
parser.add_argument("--episodes", default=100, help="Num. of test episodes")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.99  
tau = 0.001  
hidden_size = (400, 300)  

if __name__ == "__main__":

    # Create the env
    kwargs = dict()
    env = gym.make(args.env, **kwargs)
    augment = Augment(state_size=3, action_size=env.action_space.shape[0])
    num_inputs = len(augment)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = args.save_dir + args.env

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 num_inputs,
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    agent.load_checkpoint()

    agent.set_eval()

    for i in tqdm(range(20)):
        step = 0
        returns = list()
        i = i % 4
        state = torch.Tensor([env.reset(ds=i)]).to(device)
        episode_return = 0
        while True:
            
            state = augment(state[0])
            action = agent.calc_action(state, action_noise=None).to(device)

            if state.dim() == 1:
                state = state.unsqueeze(0).to(device)
            if action.dim() == 1:
                action = action.unsqueeze(0).to(device)

            q_value = agent.critic(state, action)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_return += reward
            augment.update(action[0]) # ?

            state = torch.Tensor([next_state]).to(device)

            step += 1

            if done:
                env.render()
                returns.append(episode_return)
                break

    mean = np.mean(returns)
    variance = np.var(returns)