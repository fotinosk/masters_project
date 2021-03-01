import gym

env = gym.make('LunarLanderContinuous-v2')
print(gym.spec('LunarLanderContinuous-v2').reward_threshold)