# Timor Leiderman 2021 Custom Activation Layer

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers


class Threshold(layers.Layer):
    """
    Discrete Multi Wavlelets Transform
    Input: wave_name - name of the Wavele Filters (ghm, dd2)
    TODO: add support for more wavelets
    """
    def __init__(self, algo='sure', mode='soft', **kwargs):
        super(Threshold, self).__init__(**kwargs)
        self.mode = mode.lower()
        self.algo = algo.lower()

    def build(self, input_shape):
        pass

    def call(self, inputs, training=None, mask=None):

        # DWT concat=1
        if inputs.shape[-1] == 1:
            ll_lh_hl_hh = tf.split(inputs, 2, axis=1)
            ll_hl = tf.split(ll_lh_hl_hh[0], 2, axis=2)
            lh_hh = tf.split(ll_lh_hl_hh[1], 2, axis=2)
            ll = ll_hl[0]
            lh = lh_hh[0]
            hl = ll_hl[1]
            hh = lh_hh[1]
        # DWT concat=0
        elif inputs.shape[-1] == 4:
            ll = inputs[:, :, :, 0]
            lh = inputs[:, :, :, 2]
            hl = inputs[:, :, :, 1]
            hh = inputs[:, :, :, 3]
        else:
            return None

        # calculate global threshold on the HH component
        med = tfp.stats.percentile(tf.abs(hh), 50)
        sigma = tf.math.divide(med, 0.67448975)

        if self.algo == 'sure':
            lambda_sqrt = tf.math.sqrt(2*tf.math.log(tf.constant([hh.shape[1]*hh.shape[2]], dtype=tf.float32)))
            threshold = tf.math.multiply(sigma, lambda_sqrt)
        else:
            # BayesShrink
            sigma_square = tf.math.square(sigma)
            var = tf.experimental.numpy.var(hh)
            var_square = tf.math.square(var)
            denominator = tf.math.sqrt(tf.maximum(tf.math.subtract(var_square, sigma_square), 0))
            threshold = sigma_square / denominator

        if self.mode == 'hard':
            cond = tf.math.less(hh, threshold)
            mask = tf.where(cond, tf.zeros(tf.shape(hh)), tf.ones(tf.shape(hh)))
            hh_new = tf.multiply(hh, mask)
        else:
            hh_new = tfp.math.soft_threshold(hh, threshold)
        # concat everything back to one image
        if inputs.shape[-1] == 1:
            x = tf.concat([tf.concat([ll, lh], axis=1), tf.concat([hl, hh_new], axis=1)], axis=2)
        else:
            x = tf.concat([ll, lh, hl, hh_new], axis=-1)
        return x


if __name__ == "__main__":
    pass

    import cv2
    from tensorflow.keras import Model
    from tensorflow_wavelets.Layers import DWT
    from tensorflow_wavelets.utils.cast import *
    import numpy as np
    from tensorflow_wavelets.utils.mse import mse

    img = cv2.imread("../../../Development/input/LennaGrey.png", 0)
    inputs = np.expand_dims(img, axis=0)
    inputs = np.expand_dims(inputs, axis=-1)
    # inputs = tf.cast(inputs, dtype=tf.float32)

    _, h, w, c = inputs.shape
    x_inp = layers.Input(shape=(h, w, c))
    x = DWT.DWT(name="db2", concat=1)(x_inp)
    x = Threshold(algo='sure', mode='soft')(x)
    x = DWT.IDWT(name="db2", concat=1)(x)
    model = Model(x_inp, x, name="mymodel")
    model.run_eagerly = True
    model.summary()

    #
    out = model.predict(inputs)
    print(mse(img, out[0, ..., 0]))
    # cv2.imshow("orig", out[0, ..., 0].astype("uint8"))
    # cv2.waitKey(0)
