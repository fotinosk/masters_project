from sac import SAC_Trainer

import torch
import gym
import numpy as np
from utils import ReplayBuffer
from gym_pomdp_wrappers import MuJoCoHistoryEnv
import matplotlib.pyplot as plt

def plot(rewards):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2.png')
    plt.show()


def test():
    replay_buffer_size = 1e6
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # choose env
    # ENV = ['Reacher', 'Pendulum-v0', 'HalfCheetah-v2'][2]
    ENV = 'Lunar Lander'
    if ENV == 'Lunar Lander':
        train_ENV = 'sticky-im-train-v0'
        test_ENV = 'sticky-im-test-v0'
        env = MuJoCoHistoryEnv(test_ENV, hist_len=20)
        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]
        action_range=1.
    else:
        env = NormalizedActions(gym.make(ENV))
        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]
        action_range=1.

    max_steps   = 500
    DETERMINISTIC=False
    hidden_dim = 512
    model_path = f'./models/{train_ENV}/'

    sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range, state_dim=state_dim, action_dim=action_dim)

    sac_trainer.load_model(model_path)
    for eps in range(20):
        state =  env.reset(eps%11)
        episode_reward = 0

        for step in range(max_steps):
            action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
            next_state, reward, done, _ = env.step(action)
            env.render()   

            episode_reward += reward
            state=next_state

            if done:
                break

        print('Episode: ', eps, '| Episode Reward: ', episode_reward)

if __name__ == '__main__':
    test()