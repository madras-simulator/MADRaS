"""The datatype macro for MADRaS.

For compatibility with a new RL framework, define intX and floatX 
appropriately for use in envs/observation_manager.py and other scripts.
For example, for tensorflow based RL frameworks:


import tensorflow

class MadrasDatatypes(object):

    def __init__(self):
        self.floatX = np.float32
        self.intX = np.int32
        self.tf_floatX = tf.float32
        self.tf_intX = tf.int32
    def __repr__(self):
        return ("<float:%s ,tf_float:%s ,int:%s ,tf_int: %s>"
                % (self.floatX, self.tf_floatX, self.intX, self.tf_intX))

"""
import numpy as np


class MadrasDatatypes(object):
    """Data Types class for madras."""

    def __init__(self):
        """Init Function."""
        self.floatX = np.float32
        self.intX = np.int32

    def __repr__(self):
        """Redifining Object definition."""
        return "<float:%s ,int:%s>" % (self.floatX, self.intX)
