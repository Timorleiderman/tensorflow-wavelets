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
from Scripts.debug import debug_raw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def roll_pad(data, pad_len):

    # circular shift
    # This procedure (periodic extension) can create
    # undesirable artifacts at the beginning and end
    # of the subband signals, however, it is the most
    # convenient solution.
    # When the analysis and synthesis filters are exactly symmetric,
    # a different procedure (symmetric extension) can be used,
    # that avoids the artifacts associated with periodic extension
    data_roll = tf.roll(data, shift=-pad_len, axis=1)
    # zero padding
    data_roll_pad = tf.pad(data_roll,
                              [[0, 0], [pad_len, pad_len], [0, 0], [0, 0]],
                              mode='CONSTANT',
                              constant_values=0)
    return data_roll_pad


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

    if crop == 0:
        res = fix
    else:
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


def add_sub(a, b):
    add = (a + b) / math.sqrt(2)
    sub = (a - b) / math.sqrt(2)
    return add, sub


def up_sample_fir(x, fir):
    # create zero like tensor
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    zero_tensor = tf.zeros_like(x)
    # stack both tensors
    stack_rows = tf.stack([x, zero_tensor], axis=3)
    # reshape for zero insertion between the rows
    stack_rows = tf.reshape(stack_rows, shape=[-1, x.shape[1], x.shape[2]*2, x.shape[3]])

    conv = tf.nn.conv2d(
        stack_rows, fir, padding='SAME', strides=[1, 1, 1, 1],
    )
    res = tf.transpose(conv, perm=[0, 2, 1, 3])
    return res


def analysis_filter_bank2d(x, lod_row, hid_row, lod_col, hid_col):
    # parameters
    conv_type = 'same'
    h = int(x.shape[1])
    w = int(x.shape[2])
    filt_len = int(lod_row.shape[1])

    x_roll_padd = roll_pad(x, filt_len//2)

    lo_conv_ds = fir_down_sample(x_roll_padd, lod_row)
    hi_conv_ds = fir_down_sample(x_roll_padd, hid_row)

    # # crop to needed dims
    lo = circular_shift_fix_crop(lo_conv_ds, filt_len//2, h//2)
    hi = circular_shift_fix_crop(hi_conv_ds, filt_len//2, h//2)

    # next is the columns filtering
    lo_tr = tf.transpose(lo, perm=[0, 2, 1, 3])
    hi_tr = tf.transpose(hi, perm=[0, 2, 1, 3])

    lo_tr_roll_padd = roll_pad(lo_tr, filt_len//2)
    hi_tr_roll_padd = roll_pad(hi_tr, filt_len//2)

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


def synthesis_filter_bank2d(ca, cd, lor_row, hir_row, lor_col, hir_col):

    h = int(ca.shape[1])
    w = int(ca.shape[2])
    filt_len = int(lor_row.shape[1])

    ll = tf.transpose(ca, perm=[0,2,1,3])
    lh = tf.transpose(cd[0], perm=[0,2,1,3])
    hl = tf.transpose(cd[1], perm=[0,2,1,3])
    hh = tf.transpose(cd[2], perm=[0,2,1,3])

    ll_pad = tf.pad(ll,
                [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                mode='CONSTANT',
                constant_values=0)

    lh_pad = tf.pad(lh,
                [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                mode='CONSTANT',
                constant_values=0)

    hl_pad = tf.pad(hl,
                    [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                    mode='CONSTANT',
                    constant_values=0)

    hh_pad = tf.pad(hh,
                    [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                    mode='CONSTANT',
                    constant_values=0)

    ll_conv = up_sample_fir(ll_pad, lor_col)
    lh_conv = up_sample_fir(lh_pad, hir_col)
    hl_conv = up_sample_fir(hl_pad, lor_col)
    hh_conv = up_sample_fir(hh_pad, hir_col)

    ll_lh_add = tf.math.add(ll_conv, lh_conv)
    hl_hh_add = tf.math.add(hl_conv, hh_conv)

    ll_lh_crop = ll_lh_add[:, filt_len//2:-filt_len//2-2, :, :]
    hl_hh_crop = hl_hh_add[:, filt_len//2:-filt_len//2-2, :, :]

    ll_lh_fix_crop = circular_shift_fix_crop(ll_lh_crop, filt_len-2, 2*w)
    hl_hh_fix_crop = circular_shift_fix_crop(hl_hh_crop, filt_len-2, 2*w)

    ll_lh_fix_crop_roll = tf.roll(ll_lh_fix_crop, shift=1-filt_len//2, axis=1)
    hl_hh_fix_crop_roll = tf.roll(hl_hh_fix_crop, shift=1-filt_len//2, axis=1)

    ll_lh = tf.transpose(ll_lh_fix_crop_roll, perm=[0, 2, 1, 3])
    hl_hh = tf.transpose(hl_hh_fix_crop_roll, perm=[0, 2, 1, 3])

    ll_lh_pad = tf.pad(ll_lh,
                       [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                       mode='CONSTANT',
                       constant_values=0)

    hl_hh_pad = tf.pad(hl_hh,
                       [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                       mode='CONSTANT',
                       constant_values=0)

    ll_lh_conv = up_sample_fir(ll_lh_pad, lor_row)
    hl_hh_conv = up_sample_fir(hl_hh_pad, hir_row)

    ll_lh_hl_hh_add = tf.math.add(ll_lh_conv,hl_hh_conv)
    ll_lh_hl_hh_add_crop = ll_lh_hl_hh_add[:, filt_len//2:-filt_len//2-2, :, :]

    y = circular_shift_fix_crop(ll_lh_hl_hh_add_crop, filt_len-2, 2*h)
    y = tf.roll(y, 1-filt_len//2, axis=1)

    return y


def dualtreecplx2d(x, J, Faf, af):

    x = tf.cast(x, tf.float32)
    # normalizetion
    x_norm = tf.math.divide(x, 2)

    # 2 trees J+1 lists
    w = [[[[],[]] for x in range(2)] for i in range(J+1)]

    for m in range(2):
        for n in range(2):
            lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf = construct_tf_filter(Faf[m][0], Faf[m][1], Faf[n][0], Faf[n][1])
            [lo, w[0][m][n]] = analysis_filter_bank2d(x_norm, lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf)
            for j in range(1, J):
                lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf = construct_tf_filter(af[m][0], af[m][1],
                                                                                     af[n][0], af[n][1])

                [lo, w[j][m][n]] = analysis_filter_bank2d(lo, lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf)
            w[J][m][n] = lo

    for j in range(J):
        for m in range(3):

            w[j][0][0][m], w[j][1][1][m] = add_sub(w[j][0][0][m], w[j][1][1][m])
            w[j][0][1][m], w[j][1][0][m] = add_sub(w[j][0][1][m], w[j][1][0][m])

    return w


def idualtreecplx2d(w, J, Fsf, sf):

    height = int(w[0][0][0][0].shape[1]*2)
    width = int(w[0][0][0][0].shape[2]*2)

    y = tf.zeros((height, width), dtype=tf.float32)

    for j in range(J):
        for m in range(3):

            w[j][0][0][m], w[j][1][1][m] = add_sub(w[j][0][0][m], w[j][1][1][m])
            w[j][0][1][m], w[j][1][0][m] = add_sub(w[j][0][1][m], w[j][1][0][m])

    for m in range(2):
        for n in range(2):
            lo = w[J][m][n]
            for j in [x for x in range(J)][::-1]:
                lor_row_tf, hir_row_tf, lor_col_tf, hir_col_tf = construct_tf_filter(sf[m][0], sf[m][1],
                                                                                     sf[n][0], sf[n][1])
                lo = synthesis_filter_bank2d(lo, w[j][m][n], lor_row_tf, hir_row_tf, lor_col_tf, hir_col_tf)

            lor_row_tf, hir_row_tf, lor_col_tf, hir_col_tf = construct_tf_filter(Fsf[m][0], Fsf[m][1],
                                                                                 Fsf[n][0], Fsf[n][1])
            lo = synthesis_filter_bank2d(lo, w[j][m][n], lor_row_tf, hir_row_tf, lor_col_tf, hir_col_tf)
            y = tf.math.add(y, lo)

    y = tf.math.divide(y, 2)
    return y


img_grey = cv2.imread("../input/LennaGrey.png",0)
w, h = img_grey.shape

[Faf, Fsf] = filters.FSfarras()
[af, sf] = filters.duelfilt()

# cast to float32
x_f32 = tf.cast(img_grey, dtype=tf.float32)
x_f32 = tf.expand_dims(x_f32, axis=-1)
x_f32 = tf.expand_dims(x_f32, axis=0)

J = 2
w = dualtreecplx2d(x_f32, J, Faf, af)
y = idualtreecplx2d(w, J, Fsf, sf)
# debug_raw(w)

print("yesyes")
pass