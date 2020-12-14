from networks import Actor,Critic
from memory import Buffer
from rdpg import RDPG
import gym
import gym_Boeing
import torch
import matplotlib.pyplot as plt

env = gym.make('boeing-danger-v1')

hidden_dim = [80, 100]
explore_steps = 1000
batch_size = 64

memory_size = 1e5

replaybuffer = Buffer(memory_size)

agent = RDPG(replaybuffer, env.observation_space, env.action_space, hidden_dim)

rewards = []

consequtive_success = 0

while consequtive_success < 64:  # condition for well trained model
    q_loss_list = []
    policy_loss_list = []

    state = env.reset()

    episode_reward = 0

    last_action = env.action_space.sample()

    episode_state = []
    episode_action = []
    episode_last_action = []
    episode_reward = []
    episode_next_state = []
    episode_done = []

    hidden_out = (torch.zeros([1, 1, hidden_dim[0]], dtype=torch.float).cuda(), torch.zeros([1, 1, hidden_dim[0]], dtype=torch.float).cuda())

    for step in range(5001):
        hidden_in = hidden_out

        action, hidden_out = agent.actor.get_action(state, last_action, hidden_in)

        next_state, reward, done, _ = env.step(action)

        if step==0:
            ini_hidden_in = hidden_in
            ini_hidden_out = hidden_out

        episode_state.append(state)
        episode_action.append(action)
        episode_last_action.append(last_action)
        episode_reward.append(reward)
        episode_next_state.append(next_state)
        episode_done.append(done)  

        state = next_state
        last_action = action

        if len(replaybuffer) > batch_size:
            print('Now learning')
            value_loss, policy_loss = agent.update(batch_size)
            q_loss_list.append(value_loss)
            policy_loss_list.append(policy_loss)

        if done:
            plt.clf()
            if step < 4900:
                consequtive_success += 1
            else:
                consequtive_success = 0
            env.render()
            print(f"Consequtive succesful episodes: {consequtive_success}")
            break

    replaybuffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, episode_reward, episode_next_state, episode_done)
    rewards.append(reward)



