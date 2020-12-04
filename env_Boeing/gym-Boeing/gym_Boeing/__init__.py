from gym.envs.registration import register

register(
    id='boeing-safe-v0',
    entry_point='gym_Boeing.envs:BoeingSafe',
    # tags={'wrapper_config.TimeLimit.max_episode_steps': 200},
    # reward_threshold=-3.0
)

register(
    id='boeing-danger-v0',
    entry_point='gym_Boeing.envs:BoeingDanger'
)

register(
    id='normalized-danger-v0',
    entry_point='gym_Boeing.envs:NormalizedDanger'
)