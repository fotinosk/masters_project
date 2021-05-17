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

register(
    id = 'failure-train-v2',
    entry_point = 'gym_Boeing.envs:FailureMode5'
)

register(
    id = 'simple-model-v0',
    entry_point = 'gym_Boeing.envs:SimpleModel'
)

register(
    id = 'ineffective-throtle-v0',
    entry_point = 'gym_Boeing.envs:FailureMode6'
)

register(
    id = 'faultyA-train-v0',
    entry_point = 'gym_Boeing.envs:FailureMode7'
)

register(
    id = 'faultyA-test-v0',
    entry_point = 'gym_Boeing.envs:FailureMode8'
)

register(
    id = 'combined-modes-v0',
    entry_point = 'gym_Boeing.envs:FailureMode9'
)

register(
    id = 'failure-train-v3',
    entry_point = 'gym_Boeing.envs:FailureMode10'
)

register(
    id = 'failure-test-v3',
    entry_point = 'gym_Boeing.envs:FailureMode11'
)

register(
    id = 'actuation-train-v0',
    entry_point = 'gym_Boeing.envs:FailureMode12'
)

register(
    id = 'four-modes-train-v0',
    entry_point = 'gym_Boeing.envs:FailureMode13'
)

register(
    id = 'four-modes-test-v0',
    entry_point = 'gym_Boeing.envs:FailureMode14'
)


register(
    id = 'demonstration-v0',
    entry_point = 'gym_Boeing.envs:Demo'
)

register(
    id = 'demonstration-v1',
    entry_point = 'gym_Boeing.envs:Demo2'
)