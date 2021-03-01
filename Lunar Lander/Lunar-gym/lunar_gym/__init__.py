from gym.envs.registration import register 

register(id = 'original-v0', entry_point= 'lunar_gym.envs:LunarLanderContinuous')

register(id = 'partially-obs-v0', entry_point = 'lunar_gym.envs:LunarLanderContinuous_PO')