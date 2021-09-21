import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class RigreSure(layers.Layer):
    """
    Discrete Multi Wavlelets Transform
    Input: wave_name - name of the Wavele Filters (ghm, dd2)
    TODO: add support for more wavelets
    """
    def __init__(self, **kwargs):
        super(RigreSure, self).__init__(**kwargs)
        self.size = 0
        self.low = 0.5
        self.hight = 255

    def build(self, input_shape):
        # create filter matrix
        h = int(input_shape[1])
        w = int(input_shape[2])
        self.size = h*w

    def call(self, inputs, training=None, mask=None):
        if tf.math.abs(inputs) <= self.low:
            return 0
        elif tf.math.abs(inputs) > self.high:
            return inputs
        elif (self.low < tf.math.abs(inputs)) and (tf.math.abs(inputs) <= self.high):
            return inputs * self.high * (1 - self.low/inputs)/(self.high - self.low)


if __name__=="__main__":
    print("awef")