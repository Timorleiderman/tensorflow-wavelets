# Timor Leiderman AUG 2021
# import cv2
# import numpy as np
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
    """
    Discrete Multi Wavlelets Transform
    Input: wave_name - name of the Wavele Filters (ghm, dd2)
    TODO: add support for more wavelets
    """
    def __init__(self, wave_name='ghm', **kwargs):
        super(DMWT, self).__init__(**kwargs)
        self.wave_name = wave_name.lower()

    def build(self, input_shape):
        # create filter matrix
        h = int(input_shape[1])
        w = int(input_shape[2])
        if self.wave_name == 'dd2':
            w_mat = filters.dd2(h, w)
        else:
            w_mat = filters.ghm_w_mat(h, w)
        w_mat = tf.constant(w_mat, dtype=tf.float32)
        w_mat = tf.expand_dims(w_mat, axis=0)
        self.w_mat = tf.expand_dims(w_mat, axis=-1)
        # repeat if number of channels is bigger then 1
        if input_shape[-1] != 1:
            self.w_mat = tf.repeat(self.w_mat, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):
        if self.wave_name == 'dd2':
            res = analysis_filter_bank2d_dd2_mult(inputs, self.w_mat)
        else:
            res = analysis_filter_bank2d_ghm_mult(inputs, self.w_mat)
        return res


# Inverse Discrete MultiWavelet transform Layer

class IDMWT(layers.Layer):
    """
    Inverse Multi Wavelet Transform
    wave_name - name of the Wavele Filters (ghm, dd2)
    """
    def __init__(self, wave_name='ghm', **kwargs):
        super(IDMWT, self).__init__(**kwargs)
        self.wave_name = wave_name

    def build(self, input_shape):
        # create filter matrix
        h = int(input_shape[1])//2
        w = int(input_shape[2])//2
        if self.wave_name == 'dd2':
            w_mat = filters.dd2(2*h, 2*w)
        else:
            w_mat = filters.ghm_w_mat(h, w)
        w_mat = tf.constant(w_mat, dtype=tf.float32)
        # transpose for the reconstruction
        w_mat = tf.transpose(w_mat, perm=[1, 0])
        w_mat = tf.expand_dims(w_mat, axis=-1)
        self.w_mat = tf.expand_dims(w_mat, axis=0)
        # repeat if channels bigger then 1
        if input_shape[-1] != 1:
            self.w_mat = tf.repeat(self.w_mat, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):
        if self.wave_name == 'dd2':
            res = synthesis_filter_bank2d_dd2_mult(inputs, self.w_mat)
        else:
            res = synthesis_filter_bank2d_ghm_mult(inputs, self.w_mat)

        return res


if __name__ == "__main__":
    pass
    # img = cv2.imread("../input/LennaGrey.png", 0)
    # img_ex1 = np.expand_dims(img, axis=0)
    # #
    # if len(img_ex1.shape) <= 3:
    #     img_ex1 = np.expand_dims(img_ex1, axis=-1)
    #
    #
    # _, h, w, c = img_ex1.shape
    # x_inp = layers.Input(shape=(h, w, c))
    # x = DMWT("ghm")(x_inp)
    # model = Model(x_inp, x, name="mymodel")
    # model.summary()
    #
    # out = model.predict(img_ex1)
    #
    # out_l = tf_rgb_to_ndarray(out*2)
    # out1 = cast_like_matlab_uint8_2d_rgb(out_l)
    # cv2.imshow("orig", out1.astype('uint8'))
    # cv2.waitKey(0)

    # x_inp = layers.Input(shape=(28, 28, 1))
    # x = DMWT()(x_inp)
    # # x = IDMWT()(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(10, activation="softmax")(x)
    #
    # model = Model(x_inp, x, name="mymodel")
    # model.summary()
    # optimizer = SGD(lr=1e-4, momentum=0.9)
    # model.compile(loss="categorical_crossentropy",
    #               optimizer=optimizer, metrics=["accuracy"])
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # x_train = x_train.astype('float32') / 255.0
    # x_train = np.expand_dims(x_train, axis=-1)
    #
    # x_test = x_test.astype('float32') / 255.0
    # x_test = np.expand_dims(x_test, axis=-1)
    # history = model.fit(x_train, y_train,
    #                     validation_split=0.2,
    #                     epochs=40,
    #                     batch_size=32,
    #                     verbose=2,
    #                     )