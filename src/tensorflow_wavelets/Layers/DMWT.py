# Timor Leiderman AUG 2021

from tensorflow.keras import layers
from tensorflow_wavelets.utils import filters
from tensorflow_wavelets.utils.helpers import *
from tensorflow_wavelets.utils.cast import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Discrete MultiWavelet transform Layer
class DMWT(layers.Layer):
    """
    Discrete Multi Wavlelets Transform
    Input: wave_name - name of the Wavele Filters (ghm, dd2)
    TODO: add support for more wavelets
    """
    def __init__(self, wavelet_name='ghm', **kwargs):
        super(DMWT, self).__init__(**kwargs)
        self.wave_name = wavelet_name.lower()
        self.w_mat = None

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
        self.w_mat = None

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

    import cv2
    from tensorflow.keras import Model
    from tensorflow_wavelets.Layers import DWT
    from tensorflow_wavelets.Layers.Threshold import *
    from tensorflow_wavelets.utils.cast import *
    import numpy as np
    from tensorflow_wavelets.utils.mse import mse

    img = cv2.imread("../../../src/input/LennaGrey.png", 0)
    img_ex1 = np.expand_dims(img, axis=0)
    img_ex1 = np.expand_dims(img_ex1, axis=-1)

    # _, h, w, c = img_ex1.shape
    h, w, c = 512, 512, 1
    x_inp = layers.Input(shape=(h, w, c))
    x = DMWT("ghm")(x_inp)
    x = Threshold(algo='1', mode="hard")(x)
    x = IDMWT("ghm")(x)
    model = Model(x_inp, x, name="MyModel")
    model.summary()
    model.run_eagerly = True

    out = model.predict(img_ex1)
    print(mse(img, out[0, ..., 0]))
    cv2.imshow("orig", out[0, ..., 0].astype("uint8"))
    cv2.waitKey(0)

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
