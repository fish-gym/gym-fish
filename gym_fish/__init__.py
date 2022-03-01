from gym.envs.registration import register

register(
    id='fish-basic-v0',
    entry_point='gym_fish.envs:FishEnvBasic',
)
register(
    id='koi-cruising-v0',
    entry_point='gym_fish.envs:KoiCruisingEnv',
)
register(
    id='koi-cruising3d-v0',
    entry_point='gym_fish.envs:KoiCruising3dEnv',
)
register(
    id='koifinned-cruising-v0',
    entry_point='gym_fish.envs:KoiFinnedCruisingEnv',
)
register(
    id='flatfish-cruising-v0',
    entry_point='gym_fish.envs:FlatfishCruisingEnv',
)
register(
    id='eel-cruising-v0',
    entry_point='gym_fish.envs:EelCruisingEnv',
)

register(
    id='fish-path-v0',
    entry_point='gym_fish.envs:FishEnvPathBasic',
)
register(
    id='koi-path-v0',
    entry_point='gym_fish.envs:KoiPathEnv',
)
register(
    id='eel-path-v0',
    entry_point='gym_fish.envs:EelPathEnv',
)
register(
    id='flatfish-path-v0',
    entry_point='gym_fish.envs:FlatfishPathEnv',
)

register(
    id='fish-collision-avoidance-v0',
    entry_point='gym_fish.envs:FishEnvCollisionAvoidance',
)
register(
    id='fish-pose-control-v0',
    entry_point='gym_fish.envs:FishEnvPoseControl',
)
register(
    id='fish-schooling-v0',
    entry_point='gym_fish.envs:FishEnvSchooling',
)


