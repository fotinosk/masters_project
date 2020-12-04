"""
Class that receives as inputs the normal input to the system and outputs a history
to be input in the Reinforcement Learning Model.
In this way the model gets a historic data to infer what is going on in the model

Challenges and Questions:
Give both history of actions and states, or just states?
Maybe give [state, action, next_state, reward]
How will it interact with normalized enviroment?

So input = [[state, action, next_state, reward], [state, action, next_state, reward], ...]

Take and store last 40 vectors, but average them and only give 10 (or other numbers)
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trajectory(object):

    """
    Class returns [state, np.array of trajectories]
    """
    def __init__(self, state_size, action_size,memory_size=40, outpt_size=11):
        self.memory_size = memory_size
        self.outpt_size = outpt_size
        self.action_size = action_size
        self.state_size = state_size

        # self.mock = torch.zeros((1,2*action_size+state_size+1))[0]
        # self.memory = [self.mock] * self.memory_size
        self.memory = []

    def reset(self):
        self.memory = self.mock * self.memory_size

    def __call__(self, input):
        '''Call the class to get the trajectory'''
        sup_zeros = torch.zeros((1,self.action_size+self.state_size+1))[0].to(device)
        x = torch.cat((input, sup_zeros)).to(device)
        self.memory.insert(0,x)
        if len(self.memory) < self.memory_size:
            return x
        self.memory.pop()
        outp = self.average(self.memory)
        return outp

    def archive(self, result):
        """result =  [state, action, next_state, reward]"""
        [state, action, next_state, reward] = result

        next_state = torch.Tensor(next_state).to(device)
        reward = torch.Tensor([reward]).to(device)

        if len(state) != self.num_out():
            state = state[0]

        cat_res = torch.cat((state[:self.state_size], action, next_state, reward))
        self.memory[0] = cat_res

    def average(self, output):
        current = output[0]
        past = output[1:]
        increment = self.memory_size // (self.outpt_size-1)

        output = [current]
        for i in range(self.outpt_size-1):
            roi = past[i*increment:(i+1)*increment]
            avrg = sum(roi) / increment
            output.append(avrg)
        return output

    def num_out(self):
        return int(self.action_size+2*self.state_size+1)

    def sample(self):
        return self.average(self.memory)
