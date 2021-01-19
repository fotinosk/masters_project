from gym.envs.registration import register

register(
    id='boeing-safe-v0',
    entry_point='gym_Boeing.envs:BoeingSafe',
)

register(
    id='boeing-danger-v0',
    entry_point='gym_Boeing.envs:BoeingDanger'
)

register(
    id='normalized-danger-v0',
    entry_point='gym_Boeing.envs:NormalizedDanger'
)

register(
    id='boeing-danger-v1',
    entry_point='gym_Boeing.envs:FailureDanger'
)

register(
    id='boeing-danger-v2',
    entry_point='gym_Boeing.envs:EvalDanger'
)

register(
    id = 'failure-train-v0',
    entry_point='gym_Boeing.envs:FailureMode1' 
)

register(
    id = 'boeing-danger-v3',
    entry_point='gym_Boeing.envs:Longitudinal'
)

register(
    id = 'failure-test-v0',
    entry_point='gym_Boeing.envs:FailureMode2' 
)

register(
    id = 'failure-train-v1',
    entry_point = 'gym_Boeing.envs:FailureMode3'
)

register(
    id = 'failure-test-v1',
    entry_point = 'gym_Boeing.envs:FailureMode4'
)