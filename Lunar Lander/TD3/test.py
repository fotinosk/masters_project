import gym
import numpy as np
from TD3 import TD3
from PIL import Image
import lunar_gym
from gym_pomdp_wrappers import MuJoCoHistoryEnv

def test():
    trained_env_name = "mass-train-v0"
    test_env_name = "mass-test-v0"
    random_seed = 0
    n_episodes = 100
    lr = 0.001
    max_timesteps = 3000
    render = True
    save_gif = True
    
    filename = "TD3_{}_{}".format(trained_env_name, random_seed)
    filename += '_solved'
    directory = "./models/{}".format(trained_env_name)
    
    env = MuJoCoHistoryEnv(test_env_name, hist_len=20)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    
    policy.load_actor(directory, filename)

    res_dict = {i: [] for i in range(env.modes)}
    
    for ep in range(n_episodes):
        ep_reward = 0
        state = env.reset(mode=ep%env.modes)
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
                if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                input('Press ENTER to continue.')
                break

        res_dict[ep%env.modes].append(ep_reward)    
        print('Episode: {}\tReward: {}\tMode: {}\n'.format(ep, int(ep_reward),ep%env.modes))
        ep_reward = 0
        env.close() 
    for k in res_dict.keys():
        print(k, np.mean(res_dict[k]), np.std(res_dict[k]))     
                
if __name__ == '__main__':
    test()
    
    
    