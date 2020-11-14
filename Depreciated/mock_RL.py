"""
Reinforcement learning model for mock plane model
"""
from abc import ABC

from mock_model import mock
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Network(nn.Module):
    """RL network"""

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)  # fully connected layer 1
        self.fc2 = nn.Linear(input_size, nb_action)  # fully connecter layer 2

    def forward(self, state):
        """Advance the model to the next time step"""
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


class ReplayMemory(object):
    """Implement experience replay, allow for the model to learn better"""

    def __init__(self, capacity):
        self.capacity = capacity  # how many events the memory will hold
        self.memory = []

    def push(self, event):
        """Add event to memory"""

        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """Return torch variable with a memory sample"""

        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    """Implementation of the Deep Convolutional Network"""

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)  # set memory to 100,000
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # add optimizer to nn
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        """Select action given the current state"""

        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100)
        # T = 100, T is temperature, the higher the temperature, the higher the prob the the
        # right action is chosen
        action = probs.multinomial()
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """The model learns"""

        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_output = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_output + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        """Update the state"""

        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)

        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            self.last_action = action
            self.last_state = new_state
            self.last_reward = reward
            self.reward_window.append(reward)
            if len(self.reward_window) > 100:
                del self.reward_window[0]
            return action

    def score(self):
        """Return the score"""

        return sum(self.reward_window) / (len(self.reward_window)+1.)

    def save(self):
        """Save the current model"""

        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')

    def load(self):
        """Load the current model"""
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done !')
        else:
            print('no checkpoint found...')
