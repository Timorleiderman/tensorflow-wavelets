import math
import os
import cv2
import tensorflow as tf
from utils import filters
from utils.helpers import over_sample_rows
from utils.cast import tf_to_ndarray
from utils.helpers import *


def analysis_filter_bank2d_ghm(x):
    # parameters
    conv_type = 'same'
    h = int(x.shape[1])
    w = int(x.shape[2])

    ghm_fir = filters.ghm()
    lp1, lp2, hp1, hp2 = construct_tf_filter(ghm_fir[0], ghm_fir[1], ghm_fir[2], ghm_fir[3])
    filt_len = int(lp1.shape[1])
    x_os = over_sample_rows(x)
    x_pad = tf.pad(x_os,
                               [[0, 0], [filt_len, filt_len], [0, 0], [0, 0]],
                               mode='CONSTANT',
                               constant_values=0)

    lp1_ds = fir_down_sample(x_pad, lp1, filt_len-2)
    lp1_ds1 = lp1_ds[:, 0:lp1_ds.shape[1]-5:2, :, :]

    lp2_ds = fir_down_sample(x_pad, lp2, filt_len-2)
    lp2_ds1 = lp2_ds[:, 2:lp2_ds.shape[1]-3:2, :, :]

    hp1_ds = fir_down_sample(x_pad, hp1, filt_len-2)
    hp1_ds1 = hp1_ds[:, 0:lp1_ds.shape[1]-5:2, :, :]

    hp2_ds = fir_down_sample(x_pad, hp2, filt_len-2)
    hp2_ds1 = hp2_ds[:, 2:lp2_ds.shape[1]-3:2, :, :]

    aaa =tf_to_ndarray(lp1_ds1)
    aaa =tf_to_ndarray(lp2_ds1)
    aaa =tf_to_ndarray(hp1_ds1)
    aaa =tf_to_ndarray(hp2_ds1)

    print("hello")




img_grey = cv2.imread("../input/LennaGrey.png", 0)

x_f32 = tf.cast(img_grey, dtype=tf.float32)
w, h = img_grey.shape
x_f32 = tf.expand_dims(x_f32, axis=-1)
x_f32 = tf.expand_dims(x_f32, axis=0)

analysis_filter_bank2d_ghm(x_f32)


print("hey")