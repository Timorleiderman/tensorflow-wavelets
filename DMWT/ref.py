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
    x_pad = tf.pad(x_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)

    lp1_ds = fir_down_sample(x_pad, lp1, filt_len-2)
    lp1_ds1 = lp1_ds[:, 0:lp1_ds.shape[1]-5:2, :, :]

    lp2_ds = fir_down_sample(x_pad, lp2, filt_len-2)
    lp2_ds1 = lp2_ds[:, 2:lp2_ds.shape[1]-3:2, :, :]

    hp1_ds = fir_down_sample(x_pad, hp1, filt_len-2)
    hp1_ds1 = hp1_ds[:, 0:lp1_ds.shape[1]-5:2, :, :]

    hp2_ds = fir_down_sample(x_pad, hp2, filt_len-2)
    hp2_ds1 = hp2_ds[:, 2:lp2_ds.shape[1]-3:2, :, :]*(-1)

    lp1_ds1_tr = tf.transpose(lp1_ds1, perm=[0,2,1,3])
    lp2_ds1_tr = tf.transpose(lp2_ds1, perm=[0,2,1,3])
    hp1_ds1_tr = tf.transpose(hp1_ds1, perm=[0,2,1,3])
    hp2_ds1_tr = tf.transpose(hp2_ds1, perm=[0,2,1,3])

    lp1_ds1_os = over_sample_rows(lp1_ds1_tr)
    lp2_ds1_os = over_sample_rows(lp2_ds1_tr)
    hp1_ds1_os = over_sample_rows(hp1_ds1_tr)
    hp2_ds1_os = over_sample_rows(hp2_ds1_tr)

    lp1_ds1_os_pad = tf.pad(lp1_ds1_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)
    lp2_ds1_os_pad = tf.pad(lp2_ds1_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)
    hp1_ds1_os_pad = tf.pad(hp1_ds1_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)
    hp2_ds1_os_pad = tf.pad(hp2_ds1_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)

    lp1_lp1_ds = fir_down_sample(lp1_ds1_os_pad, lp1, start=filt_len-2, step=4)
    lp1_hp1_ds = fir_down_sample(lp2_ds1_os_pad, hp1, start=filt_len-2, step=4)
    hp1_lp1_ds = fir_down_sample(hp1_ds1_os_pad, lp1, start=filt_len-2, step=4)
    hp1_hp1_ds = fir_down_sample(hp2_ds1_os_pad, hp1, start=filt_len-2, step=4)

    lp1_lp1_tr = tf.transpose(lp1_lp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    lp1_hp1_tr = tf.transpose(lp1_hp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp1_lp1_tr = tf.transpose(hp1_lp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp1_hp1_tr = tf.transpose(hp1_hp1_ds[:,:-3,:,:], perm=[0,2,1,3])

    aaalp1_lp1_ds =tf_to_ndarray(lp1_lp1_tr)
    aaalp1_hp1_ds =tf_to_ndarray(lp1_hp1_tr)
    aaahp1_lp1_ds =tf_to_ndarray(hp1_lp1_tr)
    aaahp1_hp1_ds =tf_to_ndarray(hp1_hp1_tr)

    print("hello")




img_grey = cv2.imread("../input/LennaGrey.png", 0)

x_f32 = tf.cast(img_grey, dtype=tf.float32)
w, h = img_grey.shape
x_f32 = tf.expand_dims(x_f32, axis=-1)
x_f32 = tf.expand_dims(x_f32, axis=0)

analysis_filter_bank2d_ghm(x_f32)


print("hey")