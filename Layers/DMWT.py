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


class DMWT(layers.Layer):
    def __init__(self, **kwargs):
        super(DMWT, self).__init__(**kwargs)

        self.conv_type = "SAME"
        self.border_padd = "SYMMETRIC"

        ghm_fir = filters.ghm()
        self.lp1, self.lp2, self.hp1, self.hp2 = construct_tf_filter(ghm_fir[0], ghm_fir[1], ghm_fir[2], ghm_fir[3])
        self.filt_len = int(self.lp1.shape[1])

    def build(self, input_shape):
        if input_shape[-1] != 1:
            self.lp1 = tf.repeat(self.lp1, input_shape[-1], axis=-1)
            self.lp2 = tf.repeat(self.lp2, input_shape[-1], axis=-1)
            self.hp1 = tf.repeat(self.hp1, input_shape[-1], axis=-1)
            self.hp2 = tf.repeat(self.hp2, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        # oversample
        res = analysis_filter_bank2d_ghm(inputs, self.lp1, self.lp2, self.hp1, self.hp2)
        # res = [[lp1_lp1_tr, lp1_hp1_tr,hp1_lp1_tr, hp1_hp1_tr],
        #        [lp1_lp2_tr, lp1_hp2_tr,hp1_lp2_tr, hp1_hp2_tr],
        #        [lp2_lp1_tr, lp2_hp1_tr,hp2_lp1_tr, hp2_hp1_tr],
        #        [lp2_lp2_tr, lp2_hp2_tr,hp2_lp2_tr, hp2_hp2_tr],
        #        ]
        ll = tf.concat([tf.concat([res[0][0], res[1][0]], axis=2), tf.concat([res[2][0], res[3][0]], axis=2)], axis=1)
        lh = tf.concat([tf.concat([res[0][1], res[1][1]], axis=2), tf.concat([res[2][1], res[3][1]], axis=2)], axis=1)
        hl = tf.concat([tf.concat([res[0][2], res[1][2]], axis=2), tf.concat([res[2][2], res[3][2]], axis=2)], axis=1)
        hh = tf.concat([tf.concat([res[0][3], res[1][3]], axis=2), tf.concat([res[2][3], res[3][3]], axis=2)], axis=1)

        res = tf.concat([tf.concat([ll, lh], axis=2), tf.concat([hl, hh], axis=2)], axis=1)
        return res


if __name__ == "__main__":
    img = cv2.imread("../input/LennaGrey.png", 0)
    img_ex1 = np.expand_dims(img, axis=0)
    #
    if len(img_ex1.shape) <= 3:
        img_ex1 = np.expand_dims(img_ex1, axis=-1)

    _, h, w, c = img_ex1.shape
    x_inp = layers.Input(shape=(h, w, c))
    x = DMWT()(x_inp)
    model = Model(x_inp, x, name="mymodel")
    model.summary()
    out = model.predict(img_ex1)

    out_l = tf_to_ndarray(out)
    out1 = cast_like_matlab_uint8_2d(out_l)
    cv2.imshow("orig", out1.astype('uint8'))
    cv2.waitKey(0)