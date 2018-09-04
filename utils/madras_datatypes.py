"""The datatype macro for MADRaS."""
import numpy as np
import tensorflow as tf


class Madras(object):
    """Data Types class for madras."""

    def __init__(self):
        """Init Function."""
        self.floatX = np.float32
        self.intX = np.int32
        self.tf_floatX = tf.float32
        self.tf_intX = tf.int32

    def __repr__(self):
        """Redifining Object definition."""
        return "<float:%s ,tf_float:%s ,int:%s ,tf_int: %s>"\
               % (self.floatX, self.tf_floatX, self.intX, self.tf_intX)
