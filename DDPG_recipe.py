#!/usr/bin/env python
# coding: utf-8

# # Guide to Building a Deep Deterministic Policy Gradient Model

# ## Original Paper from DeepMind

# # Overview:
# 
# - Actor-Critic Architecture
# - Model-free
# - Need discrete time steps
# - Assumes the enviroment is fully observed
# 

# x: Observations (state)
# 
# a: Action
# 
# t: Reward
# 

# Agents behaviour is determined by the policy (pi)
# 
# The policy maps states into a probability distribution of actions
# 
# (Enviromet may be stochastic)
# 
# Modeled as a Markov Decision process with transition dynamics: p(s_(t+1)|s_(t),a_(t))
# 
# The return from a state is defined as the sum of discounted rewards:
# 
#     R_t = sum(gamma^(i-t) * r(s_i, a_i))
# 
#         Where gamma in [0,1]: discount factor

# ### Goal: Learn a policy that maximizes the expecter reward from the statr distribution
# 
#     J = Exp{R_1}

# Action-value function: Describes the expected return after taking an action in a given state, following the policy policy
# 
#     Q^(pi)(s_t,a_t) = Exp{R_t|s_t, a_t}
# 
# From the Bellman Equation:
# 
#     Q^(pi)(s_t,a_t) = Exp{r(s_t,a_t) + gamma * Exp{Q^(pi)(s_t+1,a_t+1)}}

# Agent's Policy: pi: S -> A
# 
# Target Policy:  mu: S <- A (if deterministic)

# Target Policy
# 
#     Q^(mu)(s_t,a_t) = Exp(r(s_t,a_t) + gamma * Q^(mu)(s_t+1, mu(s_t+1)))
# 
#     ie a_t+1 = mu(s_t+1)
# 
# Note that in the above equation the second expectation was removed since the process is now deterministic

# Greedy Policy (For the Actor):
# 
#     mu(s) = arg_max_a Q(s,a)

# ## Loss Function
# 
#     L(theta^Q) = Exp{(Q(s_t, a_t | theta^Q) - y_t)^2}
# 
#     Where y_t = r(s_t,a_t) + gamma * Q(s_t+1, mu(s_t+1) | theta^Q)

# ### theta^Q are the parameters to be optimized

# ## Implementation

# Actor:  mu(s|theta^mu)
# 
# Critic: Q(s,a)

# ### Updating Agents
# 
# Critic: Updated as dictated by the Bellman Equation
# 
# Actor: By applying the chain rule to the expected return from the start distn J wrt the actor parameters, theta^mu
# 
# 
# 
#     Grad(J) ~ Exp{Grad_(theta^mu)(Q(s,a|theta^Q))}, at s=s_t, a=mu(s_t|theta^mu)
# 
#             = Exp{Grad_(a)Q(s,a|theta^Q)) * Grad_(theta^mu)(mu(s|theta^mu))}, at s=s_t a=mu(s_t)

# Note: Non-linear function approximators don't guarantee convergence

# ## Replay Buffer
# 
# - Finite sized cache, R
# - Transitions are sampled form the enviroment according to an exploration policy
# - Transition tuple, (s_t, a_t, s_t+1, a_t+1), is stored in the Replay Buffer
# - The buffer is FIFO, and oldest samples are discarded
# - At each timestep the actor and the critic is updated by sampling a minibatch uniformly from the buffer

# Problem:    The critic network being updated is also used to the calculate the target value, making it potentially unstable
# 
# Solution:   Implementation of a Target network that uses soft target updates

# ## Target Networks
# 
# - Create a copy of the of the actor and critic Networks, Q'(s,a|theta^Q') and mu'(s|theta^mu') respectively
# - Those will now be used to calculate the target values
# - The weights of these networks are slowly updated by having them slowly track the learned networks 

# ### Target Network Updates
# 
#     theta' = tau * theta + (1-tau) * theta'
# 
# This applies to both target networks

# Problem: Different components of the observation may have different physical units, making it difficult for the network to learn effectively
# 
# Solution: Batch Normalization

# ## Batch Normalization
# 
# - Normalizes each dimension across the samples in a minibatch to have unit mean and variance
# - Keeps the running average of the mean and variance to use for nornamization during testing
# - Used on state input, all layers of the mu network, all layers of the Q network before the action input

# ## Exploration
# 
# - Exploration policy is contructed by the addition of noise to the actor policy
# 
#         mu'(s_t) = mu(s_t|theta_t^(mu)) + Noise
# 
# - OU noise is recomended
# 
# - When the training is running the policy is periodically evaluated without the exploration noise

# # Implementation Details
# 
# - lr_actor  = 10^-4
# - lr_critic = 10^-3
# - For Q, L_2 weight decay of 10^-2 was used
# - gamma = 0.99
# - tau = 0.001
# 
# 
# - The neural networs used the rectified non-linearity for all hidden layers
# - Final output layer for actor, tanh layer, to bound the actions between [-1,1]
# - Low-dim networs have 2 hidden layers with size [400,300]
# 
# 
# - Actions are included only in the 2nd hidden layer of Q
# - The final layer weights and biases of both actor and critic are initialized from a uniform distribution [-3x10^_3, 3x10^_3]
# - Other layers were initialized from uniform distributions [-1/sqrt(f), 1/sqrt(f)]
#     - f is the fan-in of the layer
# 
# 
# - Minibatch size = 64
# - Replay buffer size = 10^6
# 
# 
# - OU noise used with theta=0.15 and sigma=0.2

# ## Components
# 
# - Actor (and Target Actor)
# - Critic (and Target Critic)
# - Replay Buffer
# - Noise
# - Action Normalization (optional?) (maybe not optional since tanh is used for output layer)

# ## Processes
# 
# - Forward Propagation
# - Backward Propagation (Network Updates)
# - Batch Normalization

# # Replay Buffer

# In[1]:


import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state', 'reward'))


class ReplayBuffer(object):

    def __init__(self, size, mini_batch_size):
        self.size = size
        self.mini_batch_size = mini_batch_size
        self.memory = []
        self.position = 0

    def add(self, *args):
        """Add transition to buffer"""

        # This is a strange way to implement it, but makes it FIFO efficiently
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = int((self.position + 1) % self.size)

    def sample(self):
        """Get a minibatch from buffer"""
        return random.sample(self.memory, self.mini_batch_size)

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return "Memory buffer used for learning, takes in a tuple: ('state', 'action', 'done', 'next_state', 'reward')"


# # UO Action Noise

# In[2]:


import numpy as np


# Here there is a decision to be made between action and parameter noise
# https://openai.com/blog/better-exploration-with-parameter-noise/
# The original DDPG paper seems to suggest using action noise
# Ornstein-Uhlenbeck noise used since it is initially correlated

class ActionNoise:

    def __init__(self, mu, theta=0.15, sigma=0.2, x0=None, dt=0.05):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu  # will be initialized as a list of zeros
        self.x0 = x0
        self.dt = dt  # same as flight model

        self.reset()  # sets x_prev

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# # Fan-in Initialization

# In[3]:


import torch
import torch.nn as nn


def fan_in_init(tensor, fan_in=None):
    # Either of the above inputs works

    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    return nn.init.uniform_(tensor, -w, w)


# # Actor Network

# In[4]:


class Actor(nn.Module):

    def __init__(self, num_inputs, action_space, init_w=3e-3, hidden_1=400, hidden_2=300, init_b=3e-4):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.num_outputs = action_space.shape[0]

        # Build the architecture of the actor
        # Investigate using LayerNorm vs BatchNorm, it seems they are the same, how??
        self.fc1 = nn.Linear(num_inputs, hidden_1)
        # self.fcn1 = nn.BatchNorm1d(hidden_1)
        self.fcn1 = nn.LayerNorm(hidden_1)

        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # self.fcn2 = nn.BatchNorm1d(hidden_2)
        self.fcn2 = nn.LayerNorm(hidden_2)

        self.fc3 = nn.Linear(hidden_2, self.num_outputs)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Initialize weights
        # All layers except last are initialized with fan in method
        self.fc1.weight.data = fan_in_init(self.fc1.weight)
        self.fc1.bias.data = fan_in_init(self.fc1.bias)

        self.fc2.weight.data = fan_in_init(self.fc2.weight)
        self.fc2.bias.data = fan_in_init(self.fc2.bias)

        # The final layer weights and biases were initialized from uniform [-3e-3, 3e-3]
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_b, init_b)

    def forward(self, inputs):
        out = self.fc1(inputs)
        out = self.fcn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.fcn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.tanh(out)
        return out


# ### LayerNorm is used for now, but might consider changing in the future

# # Critic Network

# In[5]:


class Critic(nn.Module):

    def __init__(self, num_inputs, action_space, init_w=3e-3, hidden_1=400, hidden_2=300, init_b=3e-4):
        super(Critic, self).__init__()
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.num_outputs = action_space.shape[0]

        # Build the architecture of the actor
        self.fc1 = nn.Linear(num_inputs, hidden_1)
        self.fcn1 = nn.LayerNorm(hidden_1)

        self.fc2 = nn.Linear(hidden_1 + self.num_outputs, hidden_2)
        self.fcn2 = nn.LayerNorm(hidden_2)

        self.fc3 = nn.Linear(hidden_2, 1)
        self.relu = nn.ReLU()

        self.fc1.weight.data = fan_in_init(self.fc1.weight)
        self.fc1.bias.data = fan_in_init(self.fc1.bias)

        self.fc2.weight.data = fan_in_init(self.fc2.weight)
        self.fc2.bias.data = fan_in_init(self.fc2.bias)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_b, init_b)

    def forward(self, state, action):
        x = state
        x = self.fc1(x)
        x = self.fcn1(x)
        x = self.relu(x)

        x = self.fc2(torch.cat([x, action], 1))
        x = self.fcn2(x)
        x = self.relu(x)

        out = self.fc3(x)
        return out


# # Normalized Actions

# In[6]:


# Normalized actions

import gym


class NormalizedEnv(gym.ActionWrapper):

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2
        act_b = (self.action_space.high + self.action_space.low) / 2

        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b_inv = (self.action_space.high + self.action_space.low) / 2
        return act_k_inv * (action - act_b_inv)


# # Network Updates
# 
# Hard: Copies the weights of one Network to another
# 
# Soft: Partially updates the weights based in the difference in weights

# In[7]:


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# In[8]:


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# A lot of users seem to not be implementing the L2 weight decay
# 
# Should only be implemented in the ciritc
# 
# Implment in linear layers or in optimizer (?)
# 
# They seem to be more or less the same thing, so optimizer weight decay is used, which is simpler to implemente
# 
# Or is it? If it's in the optimizer, then it is not reflected in the target ---- What does this mean???

# # DDPG
# This is the agent that brings it all together

# In[9]:


import torch.nn.functional as F
import gc
import os


class DDPG(object):

    def __init__(self, num_inputs, action_space, checkpoint_dir=None):
        self.num_inputs = num_inputs
        self.num_outputs = action_space.shape[0]
        self.action_space = action_space

        # Hyperparameters
        self.lr_actor = 10e-4
        self.lr_critic = 10e-3
        self.buffer_size = 10e6
        self.batch_size = 64
        self.noise_mean = np.zeros(self.num_outputs)
        self.tau = 0.001
        self.gamma = 0.99
        self.weight_decay = 0.01

        # create actor critic networks
        self.actor = Actor(self.num_inputs, action_space)
        self.critic = Critic(self.num_inputs, action_space)

        # create target networks
        self.target_actor = Actor(self.num_inputs, action_space)
        self.target_critic = Critic(self.num_inputs, action_space)

        # ensure that the weights of the targets are the same as the actor critic
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # set up the optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic,
                                             weight_decay=self.weight_decay)

        # create replay buffer and noise
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.noise = ActionNoise(self.noise_mean)

        # Set the directory to save the models
        if checkpoint_dir is None:
            self.checkpoint_dir = "./saves/"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def update(self):
        # Update the model paremeters by sampling the memory buffer
        batch = Transition(*zip(*self.buffer.sample()))

        state_batch = torch.cat(batch.state).float()
        action_batch = torch.cat(batch.action).float()
        reward_batch = torch.cat(batch.reward).float()
        done_batch = torch.cat(batch.done).float()
        next_state_batch = torch.cat(batch.next_state).float()

        # Using the target networks calculate the actions and values
        next_action_batch = self.target_actor(next_state_batch)
        next_qs = self.target_critic(next_state_batch, next_action_batch.detach())  # Not sure why detach is used here

        # computations
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        exp_values = reward_batch + (1 - done_batch) * self.gamma * next_qs

        # critic update
        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, exp_values.detach())
        value_loss.backward()
        # DQN uses clamp step here
        self.critic_optim.step()

        # actor update
        self.actor_optim.zero_grad()
        policy_loss = - self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # update target networks
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def get_action(self, state, add_noise=True):
        # one example I found used episode decay, I couldn't find any other case where this is used, so i ignored it
        # it is used in dqn tho
        # state = torch.from_numpy(state).float()
        self.actor.eval()  # puts actor into evaluation mode, ie not training any more, this means for eg that dropout layers dont dropout etc

        with torch.no_grad():
            # torch.no_grad() impacts the autograd engine and deactivate it. It will reduce memory usage and speed up …
            mu = self.actor(state).data

        self.actor.train()  # return actor to train mode, undos eval mode

        if add_noise:
            mu += self.noise()
        # return np.clip(mu, -1 ,1)
        return mu.clamp(self.action_space.low[0], self.action_space.high[0])

    def random_action(self):
        action = np.random.uniform(-1, 1, self.num_inputs)
        return action

    def set_eval(self):
        # set all agents to evaluation mode
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    def set_train(self):
        # set all agents to training mode
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def save(self, last_time):
        save_path = self.checkpoint_dir + f'/ep{last_time}.pth.tar'
        print('Saving...')
        checkpoint = {
            'last_timestep': last_time,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.target_actor.state_dict(),
            'critic_target': self.target_critic.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'memory': self.memory
        }
        torch.save(checkpoint, save_path)
        # Garbage collection, reclaims some memory
        gc.collect()
        print(f"Model saved: {last_time},  {save_path}")

    def load(self, path=None):
        # Loads checkpoint
        if path is None:
            path = self.get_path()

        if os.path.isfile(path):
            print("Loading checkpoint...")

        checkpoint = torch.load(path)
        timestep = checkpoint['last_timestep'] + 1

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['actor_target'])
        self.target_critic.load_state_dict(checkpoint['critic_target'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        replay_buffer = checkpoint['memory']

        gc.collect()
        print('Model Loaded')
        return timestep, replay_buffer

    def get_path(self):
        # Gets the path of the latest file
        files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith("tar"))]
        path = [os.path.join(self.checkpoint_dir, file) or file in files]
        last_file = max(path, key=os.path.getctime)
        return os.path.abspath(last_file)


# # Training and Testing

# In[10]:


import gym_Boeing
import matplotlib.pyplot as plt

env = gym.make('boeing-danger-v0')
# TODO: It seems that agent doesnt work well with normalized action (why and how it works)
# env = NormalizedEnv(env)
# get_ipython().run_line_magic('matplotlib', 'auto')


# In[11]:


agent = DDPG(3, action_space=env.action_space)

# Warmup steps are added, the agent picks random actions at first when training, to encourage exploration

# In[12]:


# hyperparameters
n_test_cycles = 10
warmup = 1000

# In[13]:


import time

# set the networks in training mode
agent.set_train()

timestep = 1
rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
epoch = 0
t = 0
time_last_checkpoint = time.time()

while timestep <= 100:
    agent.noise.reset()
    epoch_return = 0.
    state = torch.Tensor([env.reset()])

    while True:
        # TODO : sampled action may not be normalized
        # if epoch == 0 and timestep < warmup:
        #     action = env.action_space.sample()
        #     action = torch.Tensor([action])
        # else:
        #      action = agent.get_action(state)
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action.numpy()[0])
        print(done, reward, _)
        timestep += 1
        epoch_return += reward

        mask = torch.Tensor([done])
        reward = torch.Tensor([reward])
        next_state = torch.Tensor([next_state])

        agent.buffer.add(state, action, mask, next_state, reward)

        state = next_state

        epoch_value_loss = 0
        epoch_policy_loss = 0

        # TODO: only update if the warmup period is over?
        if len(agent.buffer) > agent.buffer.mini_batch_size:
            value_loss, policy_loss = agent.update()

            epoch_value_loss += value_loss
            epoch_policy_loss += policy_loss

        if done:
            break

        # TODO: implement max ep len here??

    rewards.append(epoch_return)
    value_losses.append(epoch_value_loss)
    policy_losses.append(epoch_policy_loss)

    if timestep >= 10 * t:
        # One epoch has passed and it's time to present some results to the user
        print('Epoch:', epoch)
        t += 1
        test_rewards = []

        # for results to be generated, the agent is run without exploration noise
        for _ in range(n_test_cycles):
            state = torch.Tensor(env.reset())
            test_reward = 0
            while True:
                # this is a bit different form the implementation used above, although it does the same job
                # this is due to a bug that instead of returning action:[[]], returns action:[] needing for
                # the action to be reshaped
                action = agent.get_action(state, add_noise=False)
                action = action.numpy()
                action = action.reshape((2,))
                next_state, reward, done, _ = env.step(action)
                print(done, _)
                test_reward += reward
                next_state = torch.Tensor([next_state])
                state = next_state
                if done:
                    break
            test_rewards.append(test_reward)
        mean_test_rewards.append(np.mean(test_reward))

    epoch += 1

    # save model
    agent.save(timestep, agent.buffer.memory)
    env.close()
