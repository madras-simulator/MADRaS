import gym
import MADRaS
import ray
from ray.tune.registry import register_env


def env_creator(env_config):
    env = gym.make('Madras-v0')
    return env


def register_madras():
    register_env("madras_env", env_creator)