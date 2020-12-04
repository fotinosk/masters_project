import gc
import os
import torch

import torch.nn.functional as F
from torch.optim import Adam

from utils.nets import Actor, Critic


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, checkpoint_dir=None):

        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # Define the actor
        self.actor = Actor(hidden_size, num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size, num_inputs, self.action_space).to(device)

        # Define the critic
        self.critic = Critic(hidden_size, num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size, num_inputs, self.action_space).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=1e-4)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-2
                                     )

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        if checkpoint_dir is None:
            self.checkpoint_dir = "./saved_models_historic/"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def calc_action(self, state, action_noise=None):

        # x = state.to(device)
        x = state
        if type(x) != torch.Tensor:
            x = torch.stack(x).to(device)

        self.actor.eval()  
        mu = self.actor(x)
        self.actor.train()  
        mu = mu.data

        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            mu += noise

        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])
 
        return mu

    def update_params(self, batch):

        # state_batch, action_batch, next_state_batch, reward_batch = batch
        _, action_batch, _, reward_batch = torch.split(batch, [3,2,3,1], dim=-1)


        next_action_batch = self.actor_target(batch)
        next_state_action_values = self.critic_target(batch, next_action_batch.detach())

        reward_batch = reward_batch.unsqueeze(1)
        expected_values = reward_batch + self.gamma * next_state_action_values

        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(batch, self.actor(batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

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