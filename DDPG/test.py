import argparse
import logging
import os, sys
import matplotlib.pyplot as plt
import gym
import gym_Boeing
import numpy as np
import torch
from ddpg import DDPG

# Create logger
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

plt.rcParams.update({'font.size': 16})

# Parse given arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="demonstration-v1",
                    help="Env. on which the agent should be trained")
parser.add_argument("--render", default="True", help="Render the steps")
parser.add_argument("--seed", default=0, help="Random seed")
parser.add_argument("--save_dir", default="./saved_models/", help="Dir. path to load a model")
parser.add_argument("--episodes", default=100, help="Num. of test episodes")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Some parameters, which are not saved in the network files
gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)
hidden_size = (400, 300)  # size of the hidden layers (Deepmind: 400 and 300; OpenAI: 64)

if __name__ == "__main__":

    logger.info("Using device: {}".format(device))

    # Create the env
    kwargs = dict()
    env = gym.make(args.env, **kwargs)

    # Setting rnd seed for reproducibility
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use checkpoint for safe to train danger
    checkpoint_dir = args.save_dir + 'boeing-danger-v0'

    # f = open('outptu.txt', 'a')

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    agent.load_checkpoint()

    # Load the agents parameters
    agent.set_eval()

    for _ in range(args.episodes):
        step = 0
        returns = list()
        state = torch.Tensor([env.reset(2)]).to(device)
        episode_return = 0
        while True:
            action = agent.calc_action(state, action_noise=None)
            q_value = agent.critic(state, action)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_return += reward

            state = torch.Tensor([next_state]).to(device)
            step += 1

            if done:
                env.render()
                logger.info(episode_return)
                returns.append(episode_return)
                states = env.inspect()
                states_arr = np.array(states)

                time = np.arange(0, 0.05*len(states), 0.05)

                # ax1 = plt.subplot(411)
                # ax1.set_ylabel('Forward V.')
                # plt.scatter(time, states_arr[:,0], s=2)

                # limits = [(-30,35), (-10,15), (0,85)]

                ax2 = plt.subplot(311)
                ax2.set_ylabel('Vertical V.')
                ax2.set_xlim(0,100)
                ax2.set_ylim((-30,35))
                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.scatter(time, states_arr[:,1], s=2)

                ax3 = plt.subplot(312, sharex=ax2)
                ax3.set_ylabel('Pitch Rate')
                ax3.set_ylim((-10,15))
                plt.setp(ax3.get_xticklabels(), visible=False)
                plt.scatter(time, states_arr[:,2], s=2)

                ax4 = plt.subplot(313, sharex=ax2)
                ax4.set_ylabel('Pitch Angle')
                ax4.set_ylim((-5,85))
                ax4.set_xlabel('Time (s)')
                plt.scatter(time, states_arr[:,3], s=2)

                
                plt.show()

                # sys.exit()
                break

    # f.close()
    mean = np.mean(returns)
    variance = np.var(returns)
    logger.info("Score (on 100 episodes): {} +/- {}".format(mean, variance))