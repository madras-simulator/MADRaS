import numpy as np
import gym
from MADRaS.envs.gym_madras import MadrasEnv
import os
import logging
import logging.config
import sys
# logging.config.fileConfig('logging.test')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def test_madras_vanilla():
    env = MadrasEnv()
    logging.info("Testing reset...")
    obs = env.reset()
    logging.info("Initial observation: {}."
                 " Verify if the number of dimensions {} is right.".format(obs, len(obs)))
    logging.info("Testing step...")
    for t in range(20000):
        obs, r, done, _ = env.step([-1.0, 1.0, -1.0])
        logging.info("{}: reward={}, done={}".format(t, r, done))
        if done:
            env.reset()
    os.system("pkill torcs")


def test_madras_pid():
    env = MadrasEnv()
    logging.info("Testing reset...")
    obs = env.reset()
    logging.info("Initial observation: {}."
                 " Verify if the number of dimensions {} is 29.".format(obs, len(obs)))
    logging.info("Testing step...")
    for t in range(20000):
        obs, r, done, _ = env.step([0.0, 1.0])
        logging.info("{}: reward={}, done={}".format(t, r, done))
        if done:
            env.reset()
    os.system("pkill torcs")


if __name__=='__main__':
    # test_madras_vanilla()
    test_madras_pid()