import gc
import logging
import os
import torch
import numpy as np

import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

from utils.nets import Actor, Critic
from utils.utils import *


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG_LSTM(object):

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, batch_size, checkpoint_dir=None):

        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        self.batch_size = batch_size # needed for cx, hx init

        # Define the actor
        self.actor = Actor(hidden_size, num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size, num_inputs, self.action_space).to(device)

        # Define the critic
        self.critic = Critic(hidden_size, num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size, num_inputs, self.action_space).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=1e-4)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-2
                                     )  # optimizer for the critic network

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Set the directory to save the models
        if checkpoint_dir is None:
            self.checkpoint_dir = "./saved_models_ddpg_lstm/"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def calc_action(self, state, action_noise=None):
        x = state.to(device)
        self.actor.eval()  # Sets the actor in evaluation mode
        mu, _ = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])

        return mu

    def reset_lstm_hidden_state(self, done=True):
        self.actor.reset_lstm_hidden_state(done)
        self.critic.reset_lstm_hidden_state(done)

    def update_params(self, experiences):

        if len(experiences) == 0: # not enough samples
            return

        policy_loss_total = 0
        value_loss_total = 0

        target_cx = Variable(torch.zeros(self.batch_size, 300)).type(FLOAT)
        target_hx = Variable(torch.zeros(self.batch_size, 300)).type(FLOAT)

        for t in range(len(experiences)-1):
            cx = Variable(torch.zeros(self.batch_size, 300)).type(FLOAT)
            hx = Variable(torch.zeros(self.batch_size, 300)).type(FLOAT)

            state0 = np.stack((trajectory.state0 for trajectory in experiences[t]))
            action = np.stack((trajectory.action for trajectory in experiences[t]))
            reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
            state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))

            target_action, (target_hx, target_cx) = self.agent.actor_target(to_tensor(state1, volatile=True), (target_hx, target_cx))
            next_q_value, (target_hx, target_cx) = self.agent.critic_target([to_tensor(state1, volatile=False),target_action], (target_hx, target_cx))

            target_q = to_tensor(reward) + self.discount*next_q_value
            current_q, (hx, cx)= self.agent.critic([ to_tensor(state0), to_tensor(action)], (hx, cx))


            value_loss = F.mse_loss(current_q, target_q)
            value_loss /= len(experiences) # divide by trajectory length
            value_loss_total += value_loss

            action, (hx, cx) = self.actor(to_tensor(state0), (hx, cx))
            policy_loss = -self.critic([to_tensor(state0), action], (hx, cx))[0]
            policy_loss /= len(experiences) # divide by trajectory length
            policy_loss_total += policy_loss.mean()     

            self.critic.zero_grad()
            self.actor.zero_grad()       

            policy_loss = policy_loss.mean()
            value_loss.backward(retain_graph=True)
            policy_loss.backward()
            self.critic_optim.step()
            self.actor_optim.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, last_timestep, replay_buffer):
        checkpoint_name = self.checkpoint_dir + '/ep_{}.pth.tar'.format(last_timestep)
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': replay_buffer,
        }
        torch.save(checkpoint, checkpoint_name)
        gc.collect()

    def get_path_of_latest_file(self):
        files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith(".tar"))]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_network(self, name):
        if name == 'Actor':
            return self.actor
        elif name == 'Critic':
            return self.critic
        else:
            raise NameError('name \'{}\' is not defined as a network'.format(name))