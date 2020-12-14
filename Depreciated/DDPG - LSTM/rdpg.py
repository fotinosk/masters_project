import torch
import torch.optim as optim
import torch.nn as nn
from memory import Buffer
from networks import Actor, Critic
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



class RDPG:

    def __init__(self, buffer, state_space, action_space, hidden_dims):

        self.buffer = buffer
        self.hidden_dims = hidden_dims

        self.actor  = Actor(state_space, action_space, hidden_dims)
        self.critic = Critic(state_space, action_space, hidden_dims)

        self.target_critic = Critic(state_space, action_space, hidden_dims)
        self.target_actor  = Actor(state_space, action_space, hidden_dims)

        # Define the optimizers for both networks
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=1e-4) 
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)

        # Make sure both targets are with the same weight
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.q_criterion = nn.MSELoss()

    def update(self, batch_size, gamma=0.99, tau=0.01):

        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.buffer.sample(batch_size)

        state       = torch.FloatTensor(state).to(device)
        next_state  = torch.FloatTensor(next_state).to(device)
        action      = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward      = torch.FloatTensor(reward).unsqueeze(-1).to(device)  
        done        = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        predict_q, _        = self.critic(state, action, last_action, hidden_in)
        new_action, _       = self.actor.evaluate(state, action, last_action, hidden_in) 
        new_next_action, _  = self.target_actor.evaluate(next_state, action, hidden_out)
        predict_target_q, _ = self.target_critic(next_state, new_next_action, action, hidden_out)
        predict_new_q, _    = self.critic(state, new_action, last_action, hidden_in)
        target_q, _         = reward + (1-done)*gamma*predict_new_q

        value_loss  = self.q_criterion(predict_q, target_q.detach())
        policy_loss = - torch.mean(predict_new_q)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, tau)
        soft_update(self.target_critic, self.critic, tau)

        return value_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()




