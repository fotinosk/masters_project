from ddpg import DDPG
import gym
import gym_Boeing

env = gym.make("boeing-danger-v0")

agent = DDPG(0.99,
                 0.001,
                 [400,300],
                 env.observation_space.shape[0],
                 env.action_space)

print(agent.actor)
print(agent.critic)
