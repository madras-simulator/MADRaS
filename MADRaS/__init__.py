"""Env Registration."""
from gym.envs.registration import register

register(
    id='Madras-v0',
    entry_point='MADRaS.envs:MadrasEnv',
    max_episode_steps=100,
    reward_threshold=25.0,
)
