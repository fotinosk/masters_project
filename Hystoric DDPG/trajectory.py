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

class Trajectory:

    """
    Class returns [state, np.array of trajectories]
    """

    def __init__(self, memory_size=40, outpt_size=11):
        self.memory_size = memory_size
        self.outpt_size = outpt_size
        self.mock = [torch.zeros((1,3))[0],torch.zeros((1,2))[0], torch.zeros((1,3))[0], 0]
        self.memory = [self.mock] * self.memory_size

    def reset(self):
        self.memory = self.mock * self.memory_size

    def __call__(self, input):
        '''Call the class to get the trajectory'''
        x = [input,torch.zeros((1,2))[0], torch.zeros((1,3))[0], 0]
        # self.memory[0] = x
        self.memory.insert(0,x)
        self.memory.pop()
        outp = self.average(self.memory)
        return outp

    def archive(self, result):
        """result =  [state, action, next_state, reward]"""
        self.memory[0] = result

    def average(self, output):
        current = output[0]
        past = output[1:]
        increment = self.memory_size // (self.outpt_size-1)

        output = [current]
        for i in range(self.outpt_size-1):
            roi = past[i*increment:(i+1)*increment]
            avrg = [sum(x)/increment for x in list(zip(*roi))]
            output.append(avrg)
        return output


# a = Trajectory()

# for i in range(50):
#     a(torch.tensor([1,2,4]))
#     a.archive([torch.tensor([1,2,4]), torch.tensor([i,2]), torch.tensor([2,2,4]), i])
# print(a(torch.tensor([1,2,5])))
