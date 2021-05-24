from utils import ReplayBuffer
from sac import SAC_Trainer

import torch
import gym
import numpy as np
from gym_pomdp_wrappers import MuJoCoHistoryEnv
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def plot(rewards):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2.png')
    # plt.show()

def train():
    replay_buffer_size = 1e6
    replay_buffer = ReplayBuffer(replay_buffer_size)

    ENV = 'mass-train-v0'
    env = MuJoCoHistoryEnv(ENV, hist_len=20)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    action_range=1.

    writter = SummaryWriter(comment=ENV)
    
    # hyper-parameters for RL training
    max_episodes  = 10000
    max_steps   = 501
    frame_idx   = 0
    batch_size  = 300
    explore_steps = 0  # for random action sampling in the beginning of training
    update_itr = 1
    AUTO_ENTROPY=True
    DETERMINISTIC=False
    hidden_dim = 512
    rewards     = []
    model_path = f'./models2/{ENV}/'
    solved_reward = 200

    log_f = open(f"{ENV}_ep_rewards.txt","w+")

    sac_trainer=SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range, state_dim=state_dim, action_dim=action_dim)

    for eps in range(max_episodes):
        state =  env.reset()
        episode_reward = 0
             
        for step in range(max_steps):
            if frame_idx > explore_steps:
                action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
            else:
                action = sac_trainer.policy_net.sample_action()

            next_state, reward, done, _ = env.step(action)     
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            frame_idx += 1
                   
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    _ = sac_trainer.update(batch_size, reward_scale=1., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)

            if done:
                break
        
        log_f.write('{},{}\n'.format(eps, episode_reward))
        writter.add_scalar('Episode Reward', episode_reward, eps)
        writter.flush()
        log_f.flush()

        if eps % 200 == 0 and eps>0: # plot and model saving interval
            # plot(rewards)
            # np.save('rewards_{ENV}', rewards)
            sac_trainer.save_model(model_path)
            
        print('Episode: ', eps, '| Episode Reward: ', episode_reward)
        rewards.append(episode_reward)

        if np.mean(rewards[-5:]) > solved_reward:
            print('######## Solved! ########')
            sac_trainer.save_model(model_path)
            log_f.close()
            break

    sac_trainer.save_model(model_path)

if __name__ == '__main__':
    train()