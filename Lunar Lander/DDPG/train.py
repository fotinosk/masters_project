import argparse
import random
import time
import sys
import gym
import lunar_gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")

args = parser.parse_args()

env = r"mass-train-v0"
save_dir = './lunar_models/'
seed = 0
timesteps = int(1e6)
batch_size = 64
replay_size = int(1e5)
gamma = 0.99
tau = 0.001
noise_stddev = 0.2
hidden_size = [400, 300]
n_test_cycles = 10

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Define the directory where to save and load models
    checkpoint_dir = save_dir + env
    filename = 'runs/run_' + datetime.datetime.now().strftime("%m%d%H%M")
    writer = SummaryWriter(filename)

    reward_threshold = gym.spec(env).reward_threshold 

    # Create the env
    kwargs = dict()
    env = gym.make(env, **kwargs)

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
    hidden_size = tuple(hidden_size)
    agent = DDPG(gamma, tau, hidden_size, env.observation_space.shape[0],
                 env.action_space, checkpoint_dir=checkpoint_dir)

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
            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            
            writer.add_scalar('Reward', reward, timestep)
            
            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, mask, next_state, reward)

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
                print(f"Timestep: {timestep-last_timestep} | Episode Reward: {epoch_return:.{0}f} | Time taken: {time.time()- t0:.{2}f}sec")
                last_timestep = timestep
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/return', epoch_return, epoch)

        if timestep >= 10000 * t:
            print('Epoch:', epoch)
            agent.save_checkpoint(timestep, memory)
            t += 1
            test_rewards = []
            runs = 0
            while True:
                runs += 1
                state = torch.Tensor([env.reset()]).to(device)
                test_reward = 0
                agent.set_eval()
                while True:
                    action = agent.calc_action(state)  # Selection without noise

                    next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                    test_reward += reward

                    next_state = torch.Tensor([next_state]).to(device)

                    state = next_state
                    if done:
                        if reward < 0: # ie -100 meaning it failed
                            runs = 0
                        break
                print(f"Evaluation run: {runs}, Reward: {test_reward}")
                test_rewards.append(test_reward)
            
                if runs == 0:
                    print('Success condition not met, resuming training')
                    agent.set_train()
                    break
                elif runs == 5 and np.mean(test_rewards) > 200:
                    print('Success condition satisfied, terminating training')
                    agent.save_checkpoint(timestep, memory)
                    sys.exit()


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
