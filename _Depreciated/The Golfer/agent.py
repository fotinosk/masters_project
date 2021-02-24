import gc
import os
import torch
import gym
import gym_Boeing

import torch.nn.functional as F
from the_golfer import Golfer, freeze_agent
from torch.optim import Adam
from ddpg_deeper import DDPG
from augment import Augment

device = torch.device("cpu")


class Agent(object):

        def __init__(self, gamma, env1, env2, action_space, hidden_size, checkpoint_dir=None):
            self.gamma = gamma
            self.action_space = action_space
            envA = gym.make(env1)
            envB = gym.make(env2) 

            gamma_ddpg = 0.99  
            tau = 0.001  
            hidden_size_d = (100, 400, 300)
            
            augment = Augment(state_size=3, action_size=envA.action_space.shape[0])
            num_inputs = len(augment)

            save_dir    = r"./saved_deep_models_failure_modes/"
            checkpoint_dirA = save_dir + env2
            checkpoint_dirB = save_dir + env1

            agentA = DDPG(gamma_ddpg, tau , hidden_size_d, num_inputs, envA.action_space, checkpoint_dir=checkpoint_dirA)
            agentB = DDPG(gamma_ddpg, tau , hidden_size_d, num_inputs, envB.action_space, checkpoint_dir=checkpoint_dirB)

            agentA.load_checkpoint()
            agentB.load_checkpoint()

            agentA.set_eval()
            agentB.set_eval()

            freeze_agent(agentA)
            freeze_agent(agentB)

            self.agent = Golfer(hidden_size, num_inputs, agentA, agentB)

        def calc_action(self, state):

            self.agent.eval()
            mu = self.agent(state)
            self.agent.train()
            mu = mu.data

            mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])

        def update_params(self, transition):
            
            state, action, reward, done, next_state = transition

            next_action = self.agent(next_state)













# Agent(1, 'failure-train-v0', 'failure-train-v1', [300, 400, 100, 200])
