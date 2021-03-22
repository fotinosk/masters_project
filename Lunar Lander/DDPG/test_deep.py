import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import gym
import lunar_gym
import numpy as np
import torch
from gym_pomdp_wrappers import MuJoCoHistoryEnv

from ddpg_deeper import DDPG


env         = input("Select Enviroment \n")
save_dir    = r"./lunar_models_deep/"
render      = True
seed        = 0
episodes    = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.99  
tau = 0.001  
hidden_size = (100, 400, 300)  

if __name__ == "__main__":

    # Create the env
    kwargs = dict()
    env = MuJoCoHistoryEnv(env, hist_len=20)

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    trained_model  = input('Select the trained model on which this unknown mode will be tested \n')
    checkpoint_dir = save_dir + trained_model

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    agent.load_checkpoint()
    agent.set_eval()

    # for i in tqdm(range(env.possibilities)):
    for i in tqdm(range(100)):
        step = 0
        returns = list()
        state = torch.Tensor([env.reset()]).to(device)
        # state = torch.Tensor([env.reset()]).to(device)
        episode_return = 0
        while True:
            
            action = agent.calc_action(state, action_noise=None).to(device)

            if state.dim() == 1:
                state = state.unsqueeze(0).to(device)
            if action.dim() == 1:
                action = action.unsqueeze(0).to(device)

            q_value = agent.critic(state, action)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            env.render()
            episode_return += reward

            state = torch.Tensor([next_state]).to(device)
            step += 1

            if done:
                print(f"Success: {reward == 100}")
                returns.append(episode_return)
                break