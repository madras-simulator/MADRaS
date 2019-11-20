import numpy as np
import warnings
import math
from copy import deepcopy


class RewardManager(object):
    """Composes the reward function from a given reward configuration."""
    def __init__(self, cfg):
        self.rewards = {}
        for key in cfg:
            try:
                exec("self.rewards[{}] = {}({})".format(key, key, cfg[key]))
            except:
                raise ValueError("Unknown reward class {}".format(key))

        if not self.rewards:
            warnings.warn("No reward function specified. Setting TorcsReward with "
                          "scale=1.0 as reward.")
            self.rewards['TorcsReward'] = TorcsReward({'scale': 1.0})

    def get_reward(self, game_config, game_state):
        reward = 0.0
        for reward_function in self.rewards.values():
            reward += reward_function.compute_reward(game_config, game_state)
        return reward

    def reset(self):
        for reward_function in self.rewards.values():
            reward_function.reset()


class MadrasReward(object):
    """Base class of MADRaS reward functions.
    Any new reward class must inherit this class and implement
    the following methods:
        - [required] compute_reward(game_config, game_state)
        - [optional] reset()
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def compute_reward(self, game_config, game_state):
        del game_config, game_state
        raise NotImplementedError("Successor class must implement this method.")

    def reset(self):
        pass


class TorcsReward(MadrasReward):
    """Vanilla reward function provided by TORCS."""
    def compute_reward(self, game_config, game_state):
        del game_config
        if not math.isnan(game_state["torcs_reward"]):
            return self.cfg["scale"] * game_state["torcs_reward"]
        else:
            return 0.0


class ProgressReward(MadrasReward):
    """Proportional to the fraction of the track length traversed in one step."""
    def __init__(self, cfg):
        self.prev_dist = 0.0
        super(ProgressReward, self).__init__(cfg)

    def compute_reward(self, game_config, game_state):
        reward = (self.cfg["scale"] * (game_state["distance_traversed"] - self.prev_dist)
                  / game_config.track_len)
        self.prev_dist = deepcopy(game_state["distance_traversed"])
        return reward

    def reset(self):
        self.prev_dist = 0.0