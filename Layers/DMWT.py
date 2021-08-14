# Timor Leiderman AUG 2021
import cv2
import numpy as np
from tensorflow.keras import layers, Model
from utils import filters
from utils.helpers import *
from utils.cast import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

# Discrete MultiWavelet transform Layer


class DMWT(layers.Layer):
    def __init__(self, **kwargs):
        super(DMWT, self).__init__(**kwargs)

        self.conv_type = "SAME"
        self.border_padd = "SYMMETRIC"
        w_mat = filters.ghm_w_mat()
        w_mat = tf.constant(w_mat, dtype=tf.float32)
        w_mat = tf.expand_dims(w_mat, axis=0)
        self.w_mat = tf.expand_dims(w_mat, axis=-1)

    def build(self, input_shape):
        if input_shape[-1] != 1:
            self.w_mat = tf.repeat(self.w_mat, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        res = analysis_filter_bank2d_ghm_mult(inputs, self.w_mat)
        return res


# Inverse Discrete MultiWavelet transform Layer


class IDMWT(layers.Layer):
    def __init__(self, **kwargs):
        super(IDMWT, self).__init__(**kwargs)

        w_mat = filters.ghm_w_mat()
        w_mat = tf.constant(w_mat, dtype=tf.float32)
        w_mat = tf.expand_dims(w_mat, axis=0)
        w_mat = tf.expand_dims(w_mat, axis=-1)
        self.w_mat = tf.transpose(w_mat, perm=[0, 2, 1, 3])

    def build(self, input_shape):
        if input_shape[-1] != 1:
            self.w_mat = tf.repeat(self.w_mat, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        res = synthesis_filter_bank2d_ghm_mult(inputs, self.w_mat)

        return res


if __name__ == "__main__":
    img = cv2.imread("../input/Lenna_orig.png", 1)
    img_ex1 = np.expand_dims(img, axis=0)
    #
    if len(img_ex1.shape) <= 3:
        img_ex1 = np.expand_dims(img_ex1, axis=-1)

    _, h, w, c = img_ex1.shape
    x_inp = layers.Input(shape=(h, w, c))
    x = DMWT()(x_inp)
    x = IDMWT()(x)
    model = Model(x_inp, x, name="mymodel")
    model.summary()
    out = model.predict(img_ex1)

    out_l = tf_rgb_to_ndarray(out)
    out1 = cast_like_matlab_uint8_2d_rgb(out_l)
    cv2.imshow("orig", out1.astype('uint8'))
    cv2.waitKey(0)