import numpy as np
import torch

class imagify:
    
    def __init__(self,action_size, state_size):
        self.action_size = action_size
        self.state_size = state_size
        self.in_length = 40
        # self.out_length = 10
        self.history = torch.zeros((self.in_length, self.action_size+self.state_size))
        # self.tracked = torch.zeros((self.in_length / self.out_length, self.action_size+self.state_size))

    def update(self, action, state):
        # should work with list, array and tensor (1d or 2d)
        x = torch.hstack((torch.squeeze(action), torch.squeeze(state)))     
        self.history = torch.vstack((self.history, x))[1:]

    def __call__(self):
        return self.history

    def reset(self):
        self.history = torch.zeros((self.in_length, self.action_size+self.state_size))