import numpy as np

class ObservationManager(object):
    """Composes the observation vector for a given observation mode."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.obs = None
        try:
            exec("self.obs = {}()".format(cfg["mode"]))
        except:
            raise ValueError("Unrecognized observation mode {}".format(mode))

    def get_obs(self, full_obs, game_config):
        if self.cfg["normalize"]:
            full_obs = self.normalize_obs(full_obs, game_config)
        return self.obs.get_obs(full_obs)

    def min_max_normalize(self, var, min, max):
        offset = (min + max) / 2.0
        scale = max - min
        return (var - offset) / scale

    def normalize_obs(self, full_obs, game_config):
        for key in self.cfg["obs_min"]:
            exec("full_obs.{} = self.min_max_normalize(full_obs.{}, self.cfg['obs_min'][{}], self.cfg['obs_max'][{}])".format(
                key, key, key, key))
        return full_obs


class MadrasObs(object):
    """Base class of MADRaS observation classes.
    Any new observation class must inherit this class and implement
    the following methods:
        - [required] get_obs(full_obs)
    """
    def get_obs(self, full_obs):
        del full_obs
        raise NotImplementedError("Successor class must implement this method.")


class TorcsObs(MadrasObs):
    """Vanilla observation mode defined in gym-torcs."""
    def get_obs(self, full_obs):
        obs = np.hstack((full_obs.angle,
                        full_obs.track,
                        full_obs.trackPos,
                        full_obs.speedX,
                        full_obs.speedY,
                        full_obs.speedZ,
                        full_obs.wheelSpinVel / 100.0,
                        full_obs.rpm))
        return obs

class SingleAgentSimpleLapObs(MadrasObs):
    def get_obs(self, full_obs):
        obs = np.hstack((full_obs.angle,
                        full_obs.track,
                        full_obs.trackPos,
                        full_obs.speedX,
                        full_obs.speedY,
                        full_obs.speedZ))
        return obs