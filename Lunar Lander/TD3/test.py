import gym
from TD3 import TD3
from PIL import Image
import lunar_gym
from gym_pomdp_wrappers import MuJoCoHistoryEnv

def test():
    trained_env_name = "mass-train-v0"
    test_env_name = "mass-test-v0"
    random_seed = 0
    n_episodes = 10
    lr = 0.001
    max_timesteps = 3000
    render = True
    save_gif = False
    
    filename = "TD3_{}_{}".format(trained_env_name, random_seed)
    filename += '_solved'
    directory = "./models/{}".format(trained_env_name)
    
    env = MuJoCoHistoryEnv(test_env_name, hist_len=20)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    
    policy.load_actor(directory, filename)
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
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
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()        
                
if __name__ == '__main__':
    test()
    
    
    