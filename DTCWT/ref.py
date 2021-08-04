import math
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import filters

from utils.write_raw import tensor_to_write_raw, write_raw
from utils.cast import tf_to_ndarray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def fir_down_sample(data, fir):
    # input tensors rank 4

    data_tr = tf.transpose(data, perm=[0, 2, 1, 3])
    conv = tf.nn.conv2d(
        data_tr, fir, padding='SAME', strides=[1, 1, 1, 1],
    )
    conv_tr = tf.transpose(conv, perm=[0, 2, 1, 3])

    # down sample
    lo_conv_ds = conv_tr[:, 0:conv_tr.shape[1]:2, :, :]
    return lo_conv_ds


def circular_shift_fix_crop(data, shift_fix, crop):

    circular_shift_fix = tf.math.add(data[:, 0:shift_fix, :, :],
                                     data[:, -shift_fix:, :, :])

    fix = tf.concat([circular_shift_fix, data[:, shift_fix:, :, :]], axis=1)

    res = fix[:, 0:crop, :, :]

    return res


def construct_tf_filter(lod_row, hid_row, lod_col, hid_col):

    filt_len = len(lod_row)

    lod_row_tf = tf.constant(lod_row[::-1])
    lod_row_tf = tf.reshape(lod_row_tf, (1, filt_len, 1, 1))

    hid_row_tf = tf.constant(hid_row[::-1])
    hid_row_tf = tf.reshape(hid_row_tf, (1, filt_len, 1, 1))

    lod_col_tf = tf.constant(lod_col[::-1])
    lod_col_tf = tf.reshape(lod_col_tf, (1, filt_len, 1, 1))

    hid_col_tf = tf.constant(hid_col[::-1])
    hid_col_tf = tf.reshape(hid_col_tf, (1, filt_len, 1, 1))

    return lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf


def analysis_filter_bank2d(x, lod_row, hid_row, lod_col, hid_col):
    # parameters
    conv_type = 'same'
    h = int(x.shape[1])
    w = int(x.shape[2])
    filt_len = int(lod_row.shape[1])

    # circular shift
    # This procedure (periodic extension) can create
    # undesirable artifacts at the beginning and end
    # of the subband signals, however, it is the most
    # convenient solution.
    # When the analysis and synthesis filters are exactly symmetric,
    # a different procedure (symmetric extension) can be used,
    # that avoids the artifacts associated with periodic extension
    # roll only rows
    x_roll = tf.roll(x, shift=-(filt_len//2), axis=1)
    # zero padding
    x_norm_roll_padd = tf.pad(x_roll,
                              [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                              mode='CONSTANT',
                              constant_values=0)

    lo_conv_ds = fir_down_sample(x_norm_roll_padd, lod_row)
    hi_conv_ds = fir_down_sample(x_norm_roll_padd, hid_row)

    # # crop to needed dims
    lo = circular_shift_fix_crop(lo_conv_ds, filt_len//2, h//2)
    hi = circular_shift_fix_crop(hi_conv_ds, filt_len//2, h//2)

    # next is the columns filtering
    lo_tr = tf.transpose(lo, perm=[0, 2, 1, 3])
    hi_tr = tf.transpose(hi, perm=[0, 2, 1, 3])

    lo_tr_roll = tf.roll(lo_tr, shift=-(filt_len//2), axis=1)
    hi_tr_roll = tf.roll(hi_tr, shift=-(filt_len//2), axis=1)

    # zero padding
    lo_tr_roll_padd = tf.pad(lo_tr_roll,
                              [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                              mode='CONSTANT',
                              constant_values=0)

    hi_tr_roll_padd = tf.pad(hi_tr_roll,
                             [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                             mode='CONSTANT',
                             constant_values=0)

    lo_lo_conv_ds = fir_down_sample(lo_tr_roll_padd, lod_col)
    lo_hi_conv_ds = fir_down_sample(lo_tr_roll_padd, hid_col)
    hi_lo_conv_ds = fir_down_sample(hi_tr_roll_padd, lod_col)
    hi_hi_conv_ds = fir_down_sample(hi_tr_roll_padd, hid_col)

    lo_lo = circular_shift_fix_crop(lo_lo_conv_ds, filt_len//2, w//2)
    lo_hi = circular_shift_fix_crop(lo_hi_conv_ds, filt_len//2, w//2)
    hi_lo = circular_shift_fix_crop(hi_lo_conv_ds, filt_len//2, w//2)
    hi_hi = circular_shift_fix_crop(hi_hi_conv_ds, filt_len//2, w//2)

    lo_lo = tf.transpose(lo_lo, perm=[0, 2, 1, 3])
    lo_hi = tf.transpose(lo_hi, perm=[0, 2, 1, 3])
    hi_lo = tf.transpose(hi_lo, perm=[0, 2, 1, 3])
    hi_hi = tf.transpose(hi_hi, perm=[0, 2, 1, 3])

    return [lo_lo, [lo_hi, hi_lo, hi_hi]]


def dualtreecplx2d(x, J, Faf, af):

    # normalizetion
    x_norm = tf.math.divide(x, 2)

    w = [[[[],[]] for x in range(2)] for i in range(J+1)]

    for m in range(2):
        for n in range(2):
            lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf = construct_tf_filter(Faf[m][0], Faf[m][1],
                                                                                 Faf[n][0], Faf[n][1])
            [lo, w[0][m][n]] = analysis_filter_bank2d(x_norm, lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf)

            for j in range(1, J):
                lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf = construct_tf_filter(af[m][0], af[m][0],
                                                                                     af[n][0], af[n][1])

                [lo, w[j][m][n]] = analysis_filter_bank2d(lo, lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf)
                aaaa = tf_to_ndarray(lo)

            w[J][m][n] = lo

    for j in range(J):
        for m in range(3):
            a = w[j][0][0][m]
            b = w[j][1][1][m]
            w[j][0][0][m] = (a + b) / math.sqrt(2)
            w[j][1][1][m] = (a - b) / math.sqrt(2)

            a = w[j][0][1][m]
            b = w[j][1][0][m]
            w[j][0][1][m] = (a + b) / math.sqrt(2)
            w[j][1][0][m] = (a - b) / math.sqrt(2)
    return w

img_grey = cv2.imread("../input/LennaGrey.png",0)
w, h = img_grey.shape

Faf, Fsf = filters.FSfarras()
af, sf = filters.duelfilt()

# cast to float32
x_f32 = tf.cast(img_grey, dtype=tf.float32)
x_f32 = tf.expand_dims(x_f32, axis=-1)
x_f32 = tf.expand_dims(x_f32, axis=0)

w = dualtreecplx2d(x_f32, 2, Faf, af)

lo11 = tf_to_ndarray(w[2][0][0])
lo12 = tf_to_ndarray(w[2][0][1])
lo21 = tf_to_ndarray(w[2][1][0])
lo22 = tf_to_ndarray(w[2][1][1])
tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\python_lo11.hex", lo11)
tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\python_lo12.hex", lo12)
tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\python_lo21.hex", lo21)
tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\python_lo22.hex", lo22)

ch_t2121 = tf_to_ndarray(w[1][0][1][0])
cv_t2122 = tf_to_ndarray(w[1][0][1][1])
cd_t2123 = tf_to_ndarray(w[1][0][1][2])

tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\python_ch_t2121.hex", ch_t2121)
tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\python_cv_t2122.hex", cv_t2122)
tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\python_cd_t2123.hex", cd_t2123)


print("yesyes")
pass