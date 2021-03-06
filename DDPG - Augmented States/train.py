import argparse
import logging
import os
import random
import time
import sys

import gym
import gym_Boeing
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
from augment import Augment
import datetime

# Parse given arguments
# gamma, tau, hidden_size, replay_size, batch_size, hidden_size are taken from the original paper
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="boeing-danger-v1",
                    help="the environment on which the agent should be trained ")
parser.add_argument("--render_train", default=False, type=bool,
                    help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=False, type=bool,
                    help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default="./saved_models_augmented_states2/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed (default: 0)")
parser.add_argument("--timesteps", default=1e6, type=int,
                    help="Num. of total timesteps of training (default: 1e6)")
parser.add_argument("--batch_size", default=64, type=int,
                    help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e5, type=int,
                    help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.2, type=int,
                    help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=2, default=[400, 300], type=tuple,
                    help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=20, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir + args.env
    filename = 'runs/run_' + datetime.datetime.now().strftime("%m%d%H%M")
    writer = SummaryWriter(filename)

    # Create the env
    kwargs = dict()
    env = gym.make(args.env, **kwargs)

    augment = Augment(state_size=3, action_size=env.action_space.shape[0])
    num_inputs = len(augment)

    # Set random seed for all used libraries where possible
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)
    agent = DDPG(args.gamma,
                 args.tau,
                 hidden_size,
                 num_inputs,
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    # Initialize replay memory
    memory = ReplayMemory(int(args.replay_size))

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(args.noise_stddev) * np.ones(nb_actions))

    # Define counters and other variables
    start_step = 0
    # timestep = start_step
    if args.load_model:
        start_step, memory = agent.load_checkpoint()
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 1
    time_last_checkpoint = time.time()

    while timestep <= args.timesteps:
        ou_noise.reset()
        epoch_return = 0

        state = torch.Tensor([env.reset()]).to(device)
        while True:
            state = augment(state[0])
            action = agent.calc_action(state, ou_noise).to(device)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            augment.update(action)

            next_aug_state = augment.mock_augment(next_state, state, action)


            # print(done, _)
            writer.add_scalar('Reward', reward, timestep)

            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)
            next_aug_state = torch.Tensor([next_aug_state]).to(device)

            memory.push(state, action, mask, next_aug_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

                writer.add_scalar('Value Loss', value_loss, timestep)
                writer.add_scalar('Policy Loss', policy_loss, timestep)

            if done:
                print(f"Timestep: {timestep} | Episode Length: {_['len']}")
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/return', epoch_return, epoch)

        if timestep >= 20000 * t:
            print('Epoch:', epoch)
            t += 1
            test_rewards = []
            runs = 0
            for i in range(args.n_test_cycles):
                runs += 1
                state = torch.Tensor([env.reset(i % 4)]).to(device)
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
                        if _['len'] > 4999:
                            runs = 0
                        break
                print(f"Evaluation run: {runs}, Reward: {test_reward}")
                test_rewards.append(test_reward)

                if runs == 20:
                    print('Success condition satisfied, terminating training')
                    agent.save_checkpoint(timestep, memory)
                    sys.exit()
                elif runs == 0:
                    print('Success condition not met, resuming training')
                    agent.set_train()
                    agent.save_checkpoint(timestep, memory)
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
