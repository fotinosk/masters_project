import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import gym
import gym_Boeing
import numpy as np
import torch

from ddpg import DDPG
from wrappers import NormalizedActions
from augment import Augment


env         = input("Select Enviroment)
save_dir    = "./saved_models_failure_modes/"
render      = True
seed        = 0
episodes    = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.99  
tau = 0.001  
hidden_size = (400, 300)  

if __name__ == "__main__":

    # Create the env
    kwargs = dict()
    env = gym.make(env, **kwargs)
    augment = Augment(state_size=3, action_size=env.action_space.shape[0])
    num_inputs = len(augment)

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    trained_model  = input('Select the trained model on which this unknown mode will be tested')
    checkpoint_dir = save_dir + trained_model

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 num_inputs,
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    agent.load_checkpoint()

    agent.set_eval()

    # for _ in tqdm(range(args.episodes)):
    for i in tqdm(range(env.possibilities)):
        step = 0
        returns = list()
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
                env.render(stack=True)
                returns.append(episode_return)
                break
    plt.savefig('wrong_B.png')