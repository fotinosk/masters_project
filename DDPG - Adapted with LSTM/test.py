import argparse
import logging
import os
from tqdm import tqdm

import gym
import gym_Boeing
import numpy as np
import torch

from ddpg import DDPG
from wrappers import NormalizedActions

# Create logger
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Parse given arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="boeing-danger-v0",
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
    # env = NormalizedActions(env)

    # Setting rnd seed for reproducibility
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use checkpoint for safe to train danger
    checkpoint_dir = args.save_dir + args.env

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

    for _ in tqdm(range(args.episodes)):
        step = 0
        returns = list()
        state = torch.Tensor([env.reset()]).to(device)
        episode_return = 0
        while True:
            # if args.render:
            #     env.render()

            action = agent.calc_action(state, action_noise=None)
            q_value = agent.critic(state, action)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_return += reward

            state = torch.Tensor([next_state]).to(device)

            # f.write(f"Action {action}, State: {state}")

            step += 1

            if done:
                env.render()
                logger.info(episode_return)
                returns.append(episode_return)
                # f.write(f"Episode return {episode_return}")
                break

    # f.close()
    mean = np.mean(returns)
    variance = np.var(returns)
    logger.info("Score (on 100 episodes): {} +/- {}".format(mean, variance))

# Todo: Actions are between 1 and -1, maybe fix