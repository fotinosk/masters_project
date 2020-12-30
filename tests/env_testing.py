import gym
import gym_Boeing
import numpy as np

env = gym.make('failure-train-v0')
pos = env.possibilities

class NormalizedActions(gym.ActionWrapper): 

    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        action = action * 2 / (high - low)
        action = np.clip(action, [-1,-1], [1,1])
        return action

    def reverse_action(self, action):
        print('reversing action')
        low  = self.action_space.low
        high = self.action_space.high
        
        action = action * (high - low) / 2 
        action = np.clip(action, low, high)
        
        return action

# env = NormalizedActions(env)

# action = np.array([1,30])
# env.reset()
# for i in range(100):
#     x,y,z,a = env.step(action)
#     print(x,y,a)

a = np.array([-.5, 0.5])
actual_space = np.array([10,200])

a *= actual_space
print(a)
