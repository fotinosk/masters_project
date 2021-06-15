from sac import SAC_Trainer

import torch
import gym
import numpy as np
from utils import ReplayBuffer
from gym_pomdp_wrappers import MuJoCoHistoryEnv
import matplotlib.pyplot as plt
from PIL import Image


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
        train_ENV = 'mass-train-v0'
        test_ENV = 'mass-test-v0'
        env = MuJoCoHistoryEnv(test_ENV, hist_len=20)
        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]
        action_range=1.
    else:
        env = NormalizedActions(gym.make(ENV))
        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]
        action_range=1.

    max_steps   = 1000
    DETERMINISTIC=False
    hidden_dim = 512
    model_path = f'./models2/{train_ENV}/'
    save_gif = True

    sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range, state_dim=state_dim, action_dim=action_dim)

    sac_trainer.load_model(model_path)

    res_dict = {i: [] for i in range(env.modes)}

    for eps in range(100):
        state =  env.reset(eps%env.modes)
        episode_reward = 0

        for step in range(max_steps):
            action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
            next_state, reward, done, _ = env.step(action)
            env.render()   

            episode_reward += reward
            state=next_state

            if save_gif:
                img = env.render(mode = 'rgb_array')
                img = Image.fromarray(img)
                img.save('./gif/{}.jpg'.format(step))

            if done:
                input('Press ENTER to continue.')
                break
        input('Press ENTER to continue.')
        
        
        res_dict[eps%env.modes].append(episode_reward)   
        print('Episode: ', eps, '| Episode Reward: ', episode_reward, ' | Mode: ', eps%env.modes, '\n')

    for k in res_dict.keys():
        print(k, np.mean(res_dict[k]), np.std(res_dict[k]))  

if __name__ == '__main__':
    test()