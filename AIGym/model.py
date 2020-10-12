import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gym


# no need for cnn and lstm

def normalized_col_initializer(weights, std=1.0):
    out = torch.randn(weights.size())

    # normalize the output
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))  # var(out) = std^2
    return out


def weight_init(m):
    """
    :param m: model
    :return: none
    """
    classname = m.__class__.__name__
    # classname shows what kind of connection we have

    if classname.find('Linear') != -1:  # ie if the connection is linear
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        m.weight.data.uniform_(w_bound, -w_bound)
        m.bias.data.fill_(0)


# making the brain

class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(num_inputs, 1)  # out has 1 dim, the value of the state V(s)
        self.actor_linear = nn.Linear(num_inputs, num_outputs)  # outputs a Q(s,a) for each possible action (ie output)

        # init weights
        self.apply(weight_init)

        # init col
        self.actor_linear.weight.data = normalized_col_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)  # no bias
        self.critic_linear.weight.data = normalized_col_initializer(self.critic_linear.weight.data, 1.)
        self.critic_linear.bias.data.fill_(0)
        """Small std for actor and large std for critic allows exploration vs exploitation"""
        self.train()

    def forward(self, inputs):
        return self.critic_linear(inputs), self.actor_linear(inputs)


# make the optimizer

class SharedAdam(optim.Adam):
    def __init__(self, params, lr=3, betas=(0.9, 0.999), eps=1e-8, weight_decay=8):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_dq'] = p.data.new().resize_as_(p.data).zero_()

    def shared_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_dq'].share_memory_()

    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'][0]
                bias_correction2 = 1 - beta2 ** state['step'][0]
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss


def train(rank, params, shared_model, optimizer):
    # desync agents
    torch.manual_seed(params.seed + rank)

    # get env
    env = gym.make('LunarLander-v2')
    env.seed(params.seed + rank)

    # create model
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    # not sure here, this is used to get images
    state = env.reset()
    env.render()
    state = torch.from_numpy(state)

    # indicates that training is done
    done = True

    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(params.num_steps):
            value, action_values = model(Variable(state.unsqueeze(0)))
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            # choose an action according to the distribution of probs generated above
            action = prob.multinomial().data  # returns 1x1 tensor
            log_prob = log_prob.gather(1, Variable(action))

            values.append(value)
            log_probs.append(log_prob)

            state, reward, done = env.step(action.numpy())

            done = (done or episode_length > params.max_episode_length)
            reward = max(min(reward, 1), - 1)

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            rewards.append(reward)

            if done:
                break

        # update shared network
        R = torch.zeros(1,1)
        if not done:
            value, _, _ = model.(Variable(state.unsqueeze(0)))
            R = value.data
        values.append(Variable(R))

        policy_loss = 0
        value_loss = 0

        R = Variable(R)

        gae = torch.zeros(1,1)

        for i in reversed(range(len(rewards))):
            # move back in time
            R = params.gamma * R + rewards[i]  # R = r_0 + r_1 * gamma + r_2 * gamma^2... + V(n) * gamma^n
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            td = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma * params.tau + td
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        # more gravity to policy loss
        (policy_loss + 0.5 * value_loss).backward()

        # prevent weights from taking extreme values
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        optimizer.step()





