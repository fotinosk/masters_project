import argparse
import logging
import os
from tqdm import tqdm

import gym
import lunar_gym
import numpy as np
import torch

from ddpg import DDPG
from gym_pomdp_wrappers import MuJoCoHistoryEnv

env = "mass-test-v0"
trained_env = "mass-train-v0"
render = True
seed = 0
save_dir = "./lunar_models_pomdp/"
episodes = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Some parameters, which are not saved in the network files
gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)
hidden_size = (400, 300)  # size of the hidden layers (Deepmind: 400 and 300; OpenAI: 64)

if __name__ == "__main__":

    checkpoint_dir = save_dir + trained_env

    # Create the env
    env = MuJoCoHistoryEnv(env, hist_len=20, history_type='pomdp')

    # Setting rnd seed for reproducibility
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use checkpoint for safe to train danger

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    agent.load_checkpoint()

    # Load the agents parameters
    agent.set_eval()

    for _ in tqdm(range(episodes)):
        step = 0
        returns = list()
        state = torch.Tensor([env.reset()]).to(device)
        episode_return = 0
        while True:
            if render:
                env.render()

            action = agent.calc_action(state, action_noise=None)
            q_value = agent.critic(state, action)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_return += reward

            state = torch.Tensor([next_state]).to(device)


            step += 1

            if done:
                # env.render()
                print(f"Episode Reward: {reward:.{0}f} | {_}")
                returns.append(episode_return)
                break

    mean = np.mean(returns)
    variance = np.var(returns)
