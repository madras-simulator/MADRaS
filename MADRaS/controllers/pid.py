"""PID Controller."""
import numpy as np


class PIDController(object):
    def __init__(self, cfg):
        self.accel_pid = PID(np.asarray(cfg['accel_pid']))
        self.steer_pid = PID(np.asarray(cfg['steer_pid']))
        self.accel_scale = cfg['accel_scale'] if cfg['accel_scale'] else 1.0
        self.steer_scale = cfg['steer_scale'] if cfg['steer_scale'] else 1.0
        self.prev_lane = 0
        self.prev_angle = 0
        self.prev_vel = 0

    def reset(self):
        self.accel_pid.reset_pid()
        self.steer_pid.reset_pid()
        self.prev_lane = 0
        self.prev_angle = 0
        self.prev_vel = 0

    def update(self, obs):
        self.prev_vel = obs.speedX
        self.prev_angle = obs.angle
        self.prev_lane = obs.trackPos

    def get_action(self, action):
        steer_error = (self.prev_angle - (self.prev_lane - action[0])
                       * self.steer_scale)
        accel_error = (action[1] - self.prev_vel) * self.accel_scale
        self.accel_pid.update_error(accel_error)
        self.steer_pid.update_error(steer_error)
        if self.accel_pid.output() < 0.0:
            brake = 1
        else:
            brake = 0
        return np.asarray([self.steer_pid.output(),
                           self.accel_pid.output(),
                           brake])


class PID(object):
    """Implementation of PID."""

    K = None
    error = None
    dK = None

    def __init__(self, arry):
        """Init Method."""
        self.K = arry
        self.error = np.array([0.0, 0.0, 0.0])
        self.dK = np.array([0.1, 0.00001, 0.000001])

    def __str__(self):
        """Redifine print."""
        return "error : p_error: %s i_error: %s d_error: %s"\
            % (self.error[0], self.error[1], self.error[2])

    def __repr__(self):
        """Redefine the object type."""
        return "<PID_object P: %s I: %s D: %s>"\
            % (self.K[0], self.K[1], self.K[2])

    def reset_pid(self):
        self.error = np.array([0.0, 0.0, 0.0])

    def update_error(self, e):
        """Update Step."""
        self.error[2] = e - self.error[0]
        self.error[0] = e
        self.error[1] += e

    def output(self):
        """Output."""
        return np.dot(self.K, self.error)
