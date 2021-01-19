import gym
import gym_Boeing
import numpy as np

env = gym.make('failure-train-v0')
env.reset(ds=1)
action = [1,0]

for i in range(100):
    env.step(action)
    print(env.observation)
env.render(block=True)


