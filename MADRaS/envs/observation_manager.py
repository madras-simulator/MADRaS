import numpy as np
from gym import spaces
import utils.madras_datatypes as md

MadrasDatatypes = md.MadrasDatatypes()

class ObservationManager(object):
    """Composes the observation vector for a given observation mode."""
    def __init__(self, cfg, vision=False):
        self.cfg = cfg
        self.obs = None
        self.vision = vision
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

    def get_state_dim(self):
        return self.obs.observation_dim

    def get_observation_space(self):
        return self.obs.observation_space(self.vision)


class MadrasObs(object):
    """Base class of MADRaS observation classes.
    Any new observation class must inherit this class and implement
    the following methods:
        - [required] get_obs(full_obs)
        - [required] observation_dim()
    """
    def get_obs(self, full_obs):
        del full_obs
        raise NotImplementedError("Successor class must implement this method.")

    @property
    def observation_dim(self):
        raise NotImplementedError("Successor class must implement this method.")

    def observation_space(self, vision):
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

    @property
    def observation_dim(self):
        return 29

    def observation_space(self, vision):
        if not vision:                             # Vision has to be set True if you need the images from the simulator 
            high = np.inf * np.ones(self.observation_dim)
            low = -high
            observation_space = spaces.Box(low, high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1.,
                             np.inf, 1., np.inf, 255],
                             dtype=MadrasDatatypes.floatX)
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0.,
                             -np.inf, 0., -np.inf, 0],
                             dtype=MadrasDatatypes.floatX)
            observation_space = spaces.Box(low=low, high=high)

        return observation_space


class SingleAgentSimpleLapObs(MadrasObs):
    def get_obs(self, full_obs):
        track = [(x if x > 0 else 0) for x in full_obs.track]
        obs = np.hstack((full_obs.angle,
                        track,
                        full_obs.trackPos,
                        full_obs.speedX,
                        full_obs.speedY,
                        full_obs.speedZ))
        return obs
    
    @property
    def observation_dim(self):
        return 24

    def observation_space(self, vision):
        if vision:
            raise NotImplementedError("Vision inputs not supported yet.")
        high = np.asarray([1] + 19*[1] + [np.inf] + 3*[np.inf],
                          dtype=MadrasDatatypes.floatX)
        low = np.asarray([-1] + 19*[0] + [-np.inf] + 3*[-np.inf],
                          dtype=MadrasDatatypes.floatX)

        observation_space = spaces.Box(low=low, high=high)

        return observation_space