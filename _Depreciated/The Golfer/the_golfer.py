import gc
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from ddpg_deeper import DDPG
from augment import Augment
import gym
import gym_Boeing

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

save_dir    = r"./saved_deep_models_failure_modes/"
A = 'faultyA-train-v0'
B = 'failure-train-v1'

envA = gym.make(A)
envB = gym.make(B)

gamma = 0.99  
tau = 0.001  
hidden_size = (100, 400, 300)

augment = Augment(state_size=3, action_size=envA.action_space.shape[0])
num_inputs = len(augment)

checkpoint_dirA = save_dir + A
checkpoint_dirB = save_dir + B

agentA = DDPG(gamma, tau , hidden_size, num_inputs, envA.action_space, checkpoint_dir=checkpoint_dirA)
agentB = DDPG(gamma, tau , hidden_size, num_inputs, envB.action_space, checkpoint_dir=checkpoint_dirB)

agentA.load_checkpoint()
agentB.load_checkpoint()

def freeze_agent(agent):
    for name, p in agent.actor.named_parameters():
        p.requires_grad = False
    for name, p in agent.critic.named_parameters():
        p.requires_grad = False

def cat_inputs(aug_state, *args):
    # args = action + q_value
    y = aug_state
    for arg in args:
        y = torch.cat((y, arg[0], arg[1]), 1)
    return y

def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

freeze_agent(agentA)
freeze_agent(agentB)

class Golfer(nn.Module):

    def __init__(self, hidden_size, state_size, agent1, agent2, checkpoint_dir = None):
        super(Golfer, self).__init__()
        self.hidden_size = hidden_size
        self.action_space = agent1.action_space
        num_outputs = self.action_space.shape[0]
        self.state_size = state_size
        self.agent1 = agent1
        self.agent2 = agent2

        # Supervisor architecture

        # Linear layer 1
        self.linear1 = nn.Linear(state_size, hidden_size[0]).to(device)
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Linear layer 2
        self.linear2 = nn.Linear(hidden_size[0] + 2 * (num_outputs+1), hidden_size[1]).to(device)
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # LSTM layer 
        self.lstm = nn.LSTM(hidden_size[1], hidden_size[2]).to(device)
        # initialize memory
        self.hidden_params = (torch.zeros(1,1,hidden_size[2]).to(device), 
                                torch.zeros(1,1,hidden_size[2]).to(device))

        # Linear layer 3
        self.linear3 = nn.Linear(hidden_size[2], hidden_size[3]).to(device)
        self.ln3 = nn.LayerNorm(hidden_size[3])

        # Output layer
        self.mu = nn.Linear(hidden_size[3], num_outputs).to(device)

        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        fan_in_uniform_init(self.linear3.weight)
        fan_in_uniform_init(self.linear3.bias)

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    # def forward(self, state, A_tuple, B_tuple):
    def forward(self, state):
        """Tuples include action q-value pairs"""

        actionA = agentA.calc_action(state.cuda()).to(device)
        actionB = agentB.calc_action(state.cuda()).to(device)

        if state.dim() == 1:
            state = state.unsqueeze(0).to(device)
        if actionA.dim() == 1:
            actionA = actionA.unsqueeze(0).to(device)
        if actionB.dim() == 1:
            actionB = actionB.unsqueeze(0).to(device)
        q_valueA = agentA.critic(state.cuda(), actionA.cuda()).to(device)
        q_valueB = agentB.critic(state.cuda(), actionB.cuda()).to(device)

        x = state
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # x = self.linear2(torch.cat((x, *A_tuple, *B_tuple), 1))
        x = self.linear2(torch.cat((x, actionA, q_valueA, actionB, q_valueB), 1))
        x = self.ln2(x)

        x, self.hidden_params = self.lstm(x.view(len(x), 1, -1), self.hidden_params)

        x = self.linear3(x)
        x = self.ln3(x)
        x = F.relu(x)

        mu = torch.tanh(self.mu(x))
        return mu 


# golfer = Golfer([300, 400, 100, 200], num_inputs, agentA, agentB)    

# env = gym.make('failure-train-v2')
# state = torch.Tensor([env.reset()]).to(device)
# state = augment(state[0]).to(device)

# print(g.forward(state))