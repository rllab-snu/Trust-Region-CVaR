from gym.envs.registration import register

register(
    id='Jackal-v0',
    entry_point='utils.jackal_env.env:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

register(
    id='Doggo-v0',
    entry_point='utils.doggo_env.env:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)
