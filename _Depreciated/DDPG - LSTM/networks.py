import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):

    def __init__(self, state_space, action_space, hidden_dims, init_weight=3e-3):
        super(Actor, self).__init__()

        self.action_space = action_space
        self.state_space = state_space
        self.num_inputs = state_space.shape[0]
        self.num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(self.num_inputs+self.num_outputs, hidden_dims[0]).to(device)
        self.lstm = nn.LSTM(hidden_dims[0], hidden_dims[0]).to(device)
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1]).to(device)
        self.linear3 = nn.Linear(hidden_dims[1], self.num_outputs).to(device)

        # weight initialization
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.linear3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.linear3.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

        # TODO: try LSTMCell to see effect on performance
        #       add layer/batch norms

    def forward(self, state, last_action, history):

        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)  

        x = torch.cat([state,last_action], -1)

        x = self.linear1(x)
        x = F.relu(x)

        x, hidden = self.lstm(x, history)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)

        x = F.tanh(x)

        x = x.permute(1,0,2)

        return x, hidden

    def evaluate(self, state, last_action, history):
        action, hidden_out = self.forward(state, last_action, hidden_in)
        return action, hidden_out

    def get_action(self, state, last_action, hidden_in, noise_scale=1.0):      

        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()

        action, hidden_out = self.forward(state, last_action, hidden_in)

        noise = Normal(0,1).sample(action.shape).cuda()

        action = action + noise

        return action.detach().cpu().numpy()[0][0], hidden_out

        
class Critic(nn.Module):

    def __init__(self, state_space, action_space, hidden_dims, init_weight=3e-3):
        super(Critic, self).__init__()
        self.hidden_dims = hidden_dims      
        self.action_space = action_space
        self.state_space = state_space
        self.num_inputs = state_space.shape[0]
        self.num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(self.num_inputs+2*self.num_outputs, hidden_dims[0])
        self.lstm = nn.LSTM(hidden_dims[0], hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], 1)

        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.linear3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.linear3.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, state, action, last_action, hidden_in):
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        
        x = torch.cat([state, action, last_action], -1) 

        x = self.linear1(x)
        x = F.relu(x)

        x, hidden = self.lstm(x, history)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)

        x = x.permute(1,0,2)

        return x, hidden
