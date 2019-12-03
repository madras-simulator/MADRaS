import numpy as np
import gym
from envs.gym_madras import MadrasEnv
import os


def test_madras_vanilla():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is right.".format(obs, len(obs)))
    print("Testing step...")
    for t in range(20000):
        obs, r, done, _ = env.step([0.0, 1.0, -1.0])
        print("{}: reward={}, done={}".format(t, r, done))
        # if done:
        if t%100 == 0:
            env.reset()
    os.system("pkill torcs")


def test_madras_pid():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is 29.".format(obs, len(obs)))
    print("Testing step...")
    for t in range(20000):
        obs, r, done, _ = env.step([0.0, 1.0])
        print("{}: reward={}, done={}".format(t, r, done))
        if done:
            env.reset()
    os.system("pkill torcs")


if __name__=='__main__':
    # test_madras_vanilla()
    test_madras_pid()