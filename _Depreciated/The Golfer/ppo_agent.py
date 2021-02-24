"""
Proximal Policy Optimization (PPO) version 1
----------------------------
2 actors and 1 critic
old policy given by old actor, which is delayed copy of actor

To run
------
python tutorial_PPO.py --train/test
"""
import math
import random
import sys

import gym
import gym_Boeing
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from augment import Augment
import datetime
from ddpg_deeper import DDPG
from the_golfer import agentA, agentB

import argparse
import time

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'combined-modes-v0'
RANDOMSEED = 2  # random seed

EP_MAX = 1000  # total number of episodes for training
EP_LEN = 5000  # total number of steps for each episode
GAMMA = 0.9  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 64  # update batchsize
A_UPDATE_STEPS = 10  # actor update steps
C_UPDATE_STEPS = 10  # critic update steps
EPS = 1e-8  # numerical residual
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization

###############################  PPO  ####################################

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(ValueNetwork, self).__init__()
        
        # Linear layer 1
        self.linear1 = nn.Linear(state_size, hidden_size[0]).to(device)
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Linear layer 2
        self.linear2 = nn.Linear(hidden_size[0] + 2 * (action_size+1), hidden_size[1]).to(device)
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Linear layer 3
        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2]).to(device)
        self.ln3 = nn.LayerNorm(hidden_size[2])

        # Value layer
        self.V = nn.Linear(hidden_size[2], 1).to(device)
        
    def forward(self, state, tupleA, tupleB):
        x = state

        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        
        # x = self.linear2(torch.cat((x, *tupleA, *tupleB), 1))
        x = self.linear2(torch.cat((x, tupleA, tupleB), 1))
        x = self.ln2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = self.ln3(x)
        x = F.relu(x)

        V = self.V(x)
        return V

        
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, num_actions, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Linear layer 1
        self.linear1 = nn.Linear(state_size, hidden_size[0]).to(device)
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Linear layer 2
        self.linear2 = nn.Linear(hidden_size[0] + 2 * (num_actions+1), hidden_size[1]).to(device)
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Linear layer 3
        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2]).to(device)
        self.ln3 = nn.LayerNorm(hidden_size[2])

        # Output layer
        self.mu = nn.Linear(hidden_size[2], num_actions).to(device)

        self.log_std = AddBias(torch.zeros(num_actions))  
        self.num_actions = num_actions
        self.action_range = action_range

    def forward(self, state, tupleA, tupleB):
        x = state
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # x = self.linear2(torch.cat((x, *tupleA, *tupleB), 1))
        x = self.linear2(torch.cat((x, tupleA, tupleB), 1))

        x = self.ln2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = self.ln3(x)
        x = F.relu(x)

        mu = torch.tanh(self.mu(x))

        zeros = torch.zeros(mu.size())
        if state.is_cuda:
            zeros = zeros.cuda()
        log_std = self.log_std(zeros)

        return mu, log_std
        
    def get_action(self, state, tupleA, tupleB, deterministic=False):
        state = state.unsqueeze(0).to(device)
        mean, log_std = self.forward(state, tupleA, tupleB)
        std = log_std.exp()
        normal = Normal(0, 1)
        z      = normal.sample() 
        if deterministic:
            action = mean
        else:
            action  = mean+std*z
        action = torch.clamp(action, -self.action_range, self.action_range)
        return action.squeeze(0)

    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return a.numpy()


class PPO(object):
    '''
    PPO class
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=[400, 300, 100], a_lr=3e-4, c_lr=3e-4):
        self.actor = PolicyNetwork(state_dim, hidden_dim, action_dim, 2.).to(device)
        self.actor_old = PolicyNetwork(state_dim, hidden_dim, action_dim, 2.).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=c_lr)
        print(self.actor, self.critic)

    def a_train(self, s, tA, tB, a, adv):
        '''
        Update policy network
        :param s: state
        :param a: action
        :param adv: advantage
        :return:
        '''  
        mu, log_std = self.actor(s, tA, tB)
        pi = Normal(mu, torch.exp(log_std))

        mu_old, log_std_old = self.actor_old(s, tA, tB)
        oldpi = Normal(mu_old, torch.exp(log_std_old))

        # ratio = torch.exp(pi.log_prob(a) - oldpi.log_prob(a))
        ratio = torch.exp(pi.log_prob(a)) / (torch.exp(oldpi.log_prob(a)) + EPS)

        surr = ratio * adv
        if METHOD['name'] == 'kl_pen':
            lam = METHOD['lam']
            kl = torch.distributions.kl.kl_divergence(oldpi, pi)
            kl_mean = kl.mean()
            aloss = -((surr - lam * kl).mean())
        else:  # clipping method, find this is better
            aloss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv))
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        Update old policy parameter
        :return: None
        '''
        for p, oldp in zip(self.actor.parameters(), self.actor_old.parameters()):
            oldp.data.copy_(p)


    def c_train(self, cumulative_r, s, tA, tB):
        '''
        Update actor network
        :param cumulative_r: cumulative reward
        :param s: state
        :return: None
        '''
        v = self.critic(s, tA, tB)
        advantage = cumulative_r - v
        closs = (advantage**2).mean()
        self.critic_optimizer.zero_grad()
        closs.backward()
        self.critic_optimizer.step()

    def cal_adv(self, s, tA, tB, cumulative_r):
        '''
        Calculate advantage
        :param s: state
        :param cumulative_r: cumulative reward
        :return: advantage
        '''
        advantage = cumulative_r - self.critic(s, tA, tB)
        return advantage.detach()

    def update(self, s, tA, tB, a, r):
        '''
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        '''
        s = torch.FloatTensor(s).to(device)     
        a = torch.FloatTensor(a).to(device) 
        r = torch.FloatTensor(r).to(device)   

        self.update_old_pi()
        adv = self.cal_adv(s, tA, tB, r)

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, tA, tB, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, tA, tB, a, adv)

        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s, tA, tB)     

    def choose_action(self, s, tA, tB, deterministic=False):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        a = self.actor.get_action(s, tA, tB, deterministic)
        return a.detach().cpu().numpy()
    
    def get_v(self, s, tA, tB):
        '''
        Compute value
        :param s: state
        :return: value
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.FloatTensor(s).to(device)  
        # return self.critic(s).detach().cpu().numpy()[0, 0]
        return self.critic(s, tA, tB).squeeze(0).detach().cpu().numpy()


    def save_model(self, path):
        torch.save(self.actor.state_dict(), path+'_actor')
        torch.save(self.critic.state_dict(), path+'_critic')
        torch.save(self.actor_old.state_dict(), path+'_actor_old')

    def load_model(self, path):
        
        self.actor.load_state_dict(torch.load(path+'_actor'))
        self.critic.load_state_dict(torch.load(path+'_critic'))
        self.actor_old.load_state_dict(torch.load(path+'_actor_old'))

        self.actor.eval()
        self.critic.eval()
        self.actor_old.eval()
        

def main():

    env = gym.make(ENV_NAME)
    pos = env.possibilities
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    filename = 'runs/ppo_run_' + datetime.datetime.now().strftime("%m%d%H%M")
    writer = SummaryWriter(filename)

    augment = Augment(state_size=3, action_size=action_dim)
    num_inputs = len(augment)

    # reproducible
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)

    ppo = PPO(num_inputs, action_dim)

    if args.train:
        all_ep_r = []
        for ep in range(EP_MAX):
            s = env.reset()
            # s = torch.Tensor([env.reset()]).to(device)
            buffer={
                'state':[],
                'action':[],
                'reward':[],
                'tA': [],
                'tB': []
            }
            ep_r = 0
            t0 = time.time()
            for t in range(EP_LEN):  # in one episode
                # env.render()
                s = augment(torch.Tensor([s]).to(device)[0])

                actionA = agentA.calc_action(s) 
                actionB = agentB.calc_action(s)
                qA = agentA.critic(s.unsqueeze(0), actionA.unsqueeze(0))
                qB = agentB.critic(s.unsqueeze(0), actionB.unsqueeze(0))

                tA = torch.cat((actionA.unsqueeze(0), qA),-1)
                tB = torch.cat((actionB.unsqueeze(0), qB),-1)

                # tA = (actionA.unsqueeze(0), agentA.critic(s.unsqueeze(0), actionA.unsqueeze(0)))
                # tB = (actionB.unsqueeze(0), agentB.critic(s.unsqueeze(0), actionB.unsqueeze(0)))

                a = ppo.choose_action(s, tA, tB)
                s_, r, done, _ = env.step(a)
                augment.update(torch.tensor(a).to(device))
                buffer['state'].append(s)
                buffer['action'].append(a)
                buffer['reward'].append(r)
                buffer['tA'].append(tA.squeeze())
                buffer['tB'].append(tB.squeeze()) 
                s_aug = augment.mock_augment(torch.tensor(s_).to(device), s, torch.tensor(a).to(device))
                s = s_
                ep_r += r

                writer.add_scalar('Reward', r, t*(ep+1))
                writer.add_scalar('Cumulative Reward', ep_r, t*(ep+1))

                # update ppo
                if (t + 1) % BATCH == 0 or t == EP_LEN - 1 or done:
                    if done:
                        v_s_=0
                    else:
                        v_s_ = ppo.get_v(s_aug, tA, tB)[0]
                    discounted_r = []
                    for r in buffer['reward'][::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    writer.add_scalar('Value', v_s_, t*(ep+1))
                    discounted_r.reverse()

                    bs = torch.stack(buffer['state']).cpu().numpy()
                    ba = np.vstack(buffer['action'])
                    br = np.array(discounted_r)[:, np.newaxis]
                    # btA = np.vstack(np.array(buffer['tA']))
                    # btB = np.vstack(np.array(buffer['tB']))
                    # btA = [torch.cat(tuple(i),1) for i in btA]
                    # btB = [torch.cat(tuple(i),1) for i in btB]
                    btA = torch.stack(buffer['tA'])
                    btB = torch.stack(buffer['tB'])

                    buffer['state'], buffer['action'], buffer['reward'], buffer['tA'], buffer['tB'] = [], [], [], [], []
                    ppo.update(bs, btA, btB, ba, br)

                if done:
                    break
            
            writer.add_scalar('Episode Return', ep_r, ep)

            if ep == 0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)

            if ep%50==0: # Evaluate Model
                
                ppo.save_model('model/ppo')

                test_rewards = []
                runs = 0
                while True:
                    runs += 1
                    state = torch.Tensor([env.reset(ds = runs % pos)]).to(device)
                    augment.reset()
                    test_reward = 0

                    while True:
                        state = augment(state[0])

                        actionA = agentA.calc_action(state) 
                        actionB = agentB.calc_action(state)
                        qA = agentA.critic(state.unsqueeze(0), actionA.unsqueeze(0))
                        qB = agentB.critic(state.unsqueeze(0), actionB.unsqueeze(0))

                        tA = torch.cat((actionA.unsqueeze(0), qA),-1)
                        tB = torch.cat((actionB.unsqueeze(0), qB),-1)

                        # TODO: add back deterministic option
                        action = ppo.choose_action(state, tA, tB, True)
                        augment.update(torch.tensor(action).to(device))

                        next_state, reward, done, _ = env.step(action)

                        test_reward += reward

                        next_state = torch.Tensor([next_state]).to(device)

                        state = next_state

                        if done:
                            print(_['len'])
                            if _['len'] > 4999:
                                runs = 0
                            break
                    
                    print(f"Evaluation run: {runs}, Reward: {test_reward}")
                    test_rewards.append(test_reward)

                    if runs >= pos * 4:
                        print('Success condition met, terminating training')
                        ppo.save_model('model/ppo')
                        sys.exit()
                    elif runs == 0:
                        print('Evaluation failed, resuming training')
                        break

            print(
                'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    ep, EP_MAX, ep_r,
                    time.time() - t0
                )
            )


        ppo.save_model('model/ppo')
        writer.close()

    if args.test:
        ppo.load_model('model/ppo')
        while e in range(20):
            s = env.reset()
            for i in range(EP_LEN):
                env.render()
                s = augment(torch.Tensor([s]).to(device)[0])
                a = ppo.choose_action(s, True)
                augment.update(torch.tensor(a).to(device))

                # print(a)
                s, r, done, _ = env.step(a)
                if done:
                    break

if __name__ == '__main__':
    # uncomment load checkpoint from other file
    main()

    