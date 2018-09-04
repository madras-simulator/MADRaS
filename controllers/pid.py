"""PID Controller."""
import numpy as np


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
        return "<PID_onject P: %s I: %s D: %s>"\
            % (self.K[0], self.K[1], self.K[2])

    def update_error(self, e):
        """Update Step."""
        self.error[2] = e - self.error[0]
        self.error[0] = e
        self.error[1] += e

    def output(self):
        """Output."""
        return np.dot(self.K, self.error)
