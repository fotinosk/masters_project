import argparse
import logging
import os
import random
import time
import sys

import gym
import gym_Boeing
import numpy as np
from numpy.core.shape_base import stack
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
from augment import Augment
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
args = parser.parse_args()

# env             = input('Select enviroment \n')
env             = 'simple-model-v0'
hidden_size     = [400,300]
noise_stddev    = 0.2
tau             = 0.001
gamma           = 0.99
replay_size     = 1e5
batch_size      = 64
timesteps       = 1e6
seed            = 0
save_dir        = r"./saved_simple/"
render_train    = False
render_eval     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Define the directory where to save and load models
    checkpoint_dir = save_dir + env
    filename = 'runs/run_' + datetime.datetime.now().strftime("%m%d%H%M")
    writer = SummaryWriter(filename)

    # Create the env
    kwargs = dict()
    env = gym.make(env, **kwargs)

    augment = Augment(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], memory_size=8, output_size=3)
    num_inputs = len(augment)
    # num_inputs = env.action_space.shape[0]

    # Set random seed for all used libraries where possible
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 num_inputs,
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    # Initialize replay memory
    memory = ReplayMemory(int(replay_size))

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(noise_stddev) * np.ones(nb_actions))

    # Define counters and other variables
    start_step = 0
    if args.load_model:
        start_step, memory = agent.load_checkpoint()
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    last_timestep = 0
    time_last_checkpoint = time.time()

    while timestep <= timesteps:
        ou_noise.reset()
        epoch_return = 0
        t0 = time.time()

        state = torch.Tensor([env.reset()]).to(device)
        while True:
            state = augment(state[0])
            action = agent.calc_action(state, ou_noise).to(device)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            augment.update(action)

            # print(action, state, reward) 

            next_aug_state = augment.mock_augment(next_state, state, action)

            writer.add_scalar('Reward', reward, timestep)

            timestep += 1
            epoch_return += reward

            state = state.unsqueeze(0).to(device)
            action = action.unsqueeze(0).to(device)
            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)
            next_aug_state = torch.Tensor([next_aug_state]).to(device)

            memory.push(state, action, mask, next_aug_state, reward)
            # memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

                writer.add_scalar('Value Loss', value_loss, timestep)
                writer.add_scalar('Policy Loss', policy_loss, timestep)

            if done:
                print(f"Timestep: {timestep-last_timestep} | Episode Reward: {epoch_return[0]:.{0}f} | Time taken: {time.time()- t0:.{2}f}sec")
                last_timestep = timestep
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/return', epoch_return, epoch)

        if timestep >= 10000 * t:
            print('Epoch:', epoch)
            t += 1
            test_rewards = []
            runs = 0
            while True:
                runs += 1
                state = torch.Tensor([env.reset()]).to(device)
                augment.reset()
                test_reward = 0
                agent.set_eval()
                while True:
                    state = augment(state[0])
                    action = agent.calc_action(state)

                    next_state, reward, done, _ = env.step(action.cpu().numpy())
                    augment.update(action)
                    test_reward += reward

                    next_aug_state = augment.mock_augment(next_state, state, action)
                    next_state = torch.Tensor([next_state]).to(device)

                    state = next_state
                    if done:
                        print(_['len'])
                        if _['len'] > 499:
                            runs = 0
                        break
                print(f"Evaluation run: {runs}, Reward: {test_reward}")
                test_rewards.append(test_reward)

                agent.save_checkpoint(timestep, memory)

                if runs == 10:
                    print('Success condition satisfied, terminating training')
                    agent.save_checkpoint(timestep, memory)
                    sys.exit()
                elif runs == 0:
                    print('Success condition not met, resuming training')
                    agent.set_train()
                    break

            mean_test_rewards.append(np.mean(test_rewards))
            print('Epoch return: ', np.mean(test_rewards))

            for name, param in agent.actor.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)


        epoch += 1

    agent.save_checkpoint(timestep, memory)
    env.close()
    writer.close()
