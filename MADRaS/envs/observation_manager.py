import numpy as np

class ObservationManager(object):
    def __init__(self, mode='TorcsObs'):
        self.obs = None
        try:
            exec("self.obs = {}()".format(mode))
        except:
            raise ValueError("Unrecognized observation mode {}".format(mode))

    def get_obs(self, full_obs):
        return self.obs.get_obs(full_obs)


class MadrasObs(object):
    def get_obs(self, full_obs):
        del full_obs
        raise NotImplementedError("Successor class must implement this method.")


class TorcsObs(MadrasObs):
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