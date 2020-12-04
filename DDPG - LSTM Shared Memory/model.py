# Importing the librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # thanks to this initialization, we have var(out) = std^2
    return out

def weights_init(m):
    classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if classname.find('Conv') != -1: # if the connection is a convolution
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros
    elif classname.find('Linear') != -1: # if the connection is a full connection
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros

def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPG(nn.Module):

    def __init__(self, num_inputs, action_space):
        super(DDPG, self).__init__()
        nump_outputs = action_space.shape[0]

        self.s_l1 = nn.Linear(num_inputs, 400)
        self.s_ln1 = nn.LayerNorm(400)

        self.lstm = nn.LSTMCell(400,400)

        # Actor
        self.a_linear1 = nn.Linear(400,300)
        self.a_ln = nn.LayerNorm(300)
        self.a_linear2 = nn.Linear(300, nump_outputs)

        # Critic
        self.c_linear1 = nn.Linear(400+nump_outputs, 300)
        self.c_ln = nn.LayerNorm(300)
        self.c_linear2 = nn.Linear(300, 1)
        
        # Weight Init
        fan_in_uniform_init(self.s_l1.weight)
        fan_in_uniform_init(self.s_l1.bias)

        fan_in_uniform_init(self.a_linear1.weight)
        fan_in_uniform_init(self.a_linear1.bias)

        fan_in_uniform_init(self.a_linear2.weight)
        fan_in_uniform_init(self.a_linear2.bias)

        fan_in_uniform_init(self.c_linear1.weight)
        fan_in_uniform_init(self.c_linear1.bias)

        fan_in_uniform_init(self.c_linear2.weight)
        fan_in_uniform_init(self.c_linear2.bias)

        nn.init.uniform_(self.a_linear2.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.a_linear2.bias, -3e-4, 3e-4)

        nn.init.uniform_(self.c_linear2.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.c_linear2.bias, -3e-4, 3e-4)

        self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros
        self.train() # setting the module in "train" mode to activate the dropouts and batchnorms

    def forward(self, inputs):
        inputs, (hx, cx) = inputs

        x = self.s_l1(inputs)
        x = self.s_ln1(x)
        x = F.relu(x)

        (hx, cx) = self.lstm(x, (hx,cx))
        x = hx

        # Actor 
        xa = self.a_linear1(x)
        xa = self.a_ln(xa)
        xa = F.relu(xa)
        xa = self.a_linear2(xa)
        action = F.tanh(xa)

        # Critic
        xc = torch.cat((x,action), 1)
        xc = self.c_linear1(xc)
        xc = self.c_ln(xc)
        xc = F.relu(xc)
        V = self.c_linear2(xc)

        return V, action, (hx, cx)