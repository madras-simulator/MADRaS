import numpy as np
import gym
from envs.gym_madras import MadrasEnv


def test_madras_vanilla():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is 29.".format(obs, len(obs)))
    print("Testing step...")
    for t in range(2000):
        obs, r, done, _ = env.step([0.0, 100.0, 0])
        print("{}: reward={}, done={}".format(t, r, done))


def test_madras_pid():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is 29.".format(obs, len(obs)))
    print("Testing step...")
    for t in range(20):
        obs, r, done, _ = env.step([0.0, 1.0])
        print("{}: reward={}, done={}".format(t, r, done))


if __name__=='__main__':
    test_madras_vanilla()
    # test_madras_pid()