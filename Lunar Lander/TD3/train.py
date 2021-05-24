from os import write
import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
from gym_pomdp_wrappers import MuJoCoHistoryEnv
from torch.utils.tensorboard import SummaryWriter, writer

def train():
    ######### Hyperparameters #########
    env_name = "inertia-train-v0"
    random_seed = 0
    log_interval = 10           # print avg reward after interval
    solved_reward = 200          # stop training if avg_reward > solved_reward
    save_episode = 500         # keep saving after n episodes
    max_episodes = 10000        # max num of episodes
    max_timesteps = 501         # max timesteps in one episode

    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise std
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter

    directory = "./models2/{}".format(env_name)  # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    ###################################
    
    # env = gym.make(env_name)
    env = MuJoCoHistoryEnv(env_name, hist_len=20)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    writer = SummaryWriter(comment=env_name)
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open(f"{env_name}_ep_rewards.txt","w+")
    
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)
            
            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            
            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break
        
        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        writer.add_scalar("Episode Reward", ep_reward, episode)
        writer.flush()
        log_f.flush()
        ep_reward = 0
        
        # if avg reward > 200 then save and stop traning:
        if (avg_reward/log_interval) >= solved_reward:
            print("########## Solved! ###########")
            name = filename + '_solved'
            policy.save(directory, name)
            log_f.close()
            break
        
        if episode > save_episode:
            policy.save(directory, filename)
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

if __name__ == '__main__':
    train()
    