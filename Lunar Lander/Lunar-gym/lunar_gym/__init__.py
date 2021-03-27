from gym.envs.registration import register 

register(id = 'original-v0', entry_point= 'lunar_gym.envs:LunarLanderContinuous')

register(id = 'partially-obs-v0', entry_point = 'lunar_gym.envs:LunarLanderContinuous_PO')

register(id='mass-train-v0', entry_point= 'lunar_gym.envs:MassModeTrain')

register(id='mass-test-v0', entry_point = 'lunar_gym.envs:MassModeTest')

register(id='mass-train-v1', entry_point = 'lunar_gym.envs:MassModeTrainPO')

register(id='inertia-train-v0', entry_point = 'lunar_gym.envs:InertiaModeTrain')

register(id='inertia-test-v0', entry_point = 'lunar_gym.envs:InertiaModeTest')

register(id='inertia-mass-train-v0', entry_point = 'lunar_gym.envs:InertiaMassTest')
