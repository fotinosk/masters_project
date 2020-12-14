import gym
import gym_Boeing

from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env


env = gym.make('normalized-danger-v0')

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if i % 100 == 0:
        env.render()
    if done:
      obs = env.reset()