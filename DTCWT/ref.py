import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import filters

from utils import write_raw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

img_grey = cv2.imread("../input/LennaGrey.png",0)

w, h = img_grey.shape


def analysis_filter_bank2d(x, lod_row, hid_row, lod_col, hid_col):

    # parameters
    conv_type = 'same'
    h, w = x.shape
    filt_len = len(lod_row)
    # cast to float32
    x_f32 = tf.cast(x, dtype=tf.float32)
    x_f32 = tf.expand_dims(x_f32, axis=-1)
    x_f32 = tf.expand_dims(x_f32, axis=0)

    lod_row_tf = tf.constant(lod_row[::-1])
    lod_row_tf = tf.reshape(lod_row_tf, (1, filt_len, 1, 1))

    # normalizetion
    x_norm = tf.math.divide(x_f32, 2)
    # for bit accuracy to matlab model
    x_norm = tf.math.ceil(x_norm)

    # circular shift
    # This procedure (periodic extension) can create
    # undesirable artifacts at the beginning and end
    # of the subband signals, however, it is the most
    # convenient solution.
    # When the analysis and synthesis filters are exactly symmetric,
    # a different procedure (symmetric extension) can be used,
    # that avoids the artifacts associated with periodic extension
    # roll only rows
    x_norm_roll = tf.roll(x_norm, shift=-(filt_len//2), axis=1)

    # zero padding
    x_norm_roll_padd = tf.pad(x_norm_roll,
                              [[0, 0], [filt_len//2, filt_len//2], [0, 0], [0, 0]],
                              mode='CONSTANT',
                              constant_values=0)

    # Low pass filter
    x_norm_roll_padd = tf.transpose(x_norm_roll_padd, perm=[0, 2, 1, 3])
    x_norm_roll_padd_lo_conv = tf.nn.conv2d(
        x_norm_roll_padd, lod_row_tf, padding='SAME', strides=[1, 1, 1, 1],
    )
    x_norm_roll_padd_lo_conv = tf.transpose(x_norm_roll_padd_lo_conv, perm=[0, 2, 1, 3])

    # down sampole
    x_norm_roll_padd_lo_conv_ds = x_norm_roll_padd_lo_conv[:, 0:x_norm_roll_padd_lo_conv.shape[1]:2, :, :]

    # circular shift fix
    circular_shift_fix = tf.math.add(x_norm_roll_padd_lo_conv_ds[:, 0:filt_len//2, :, :],
                                     x_norm_roll_padd_lo_conv_ds[:, -filt_len//2:, :, :])
    # tensorflow do not support assignment so we will crop and merge
    x_norm_roll_padd_lo_conv_ds_fix = tf.concat([circular_shift_fix, x_norm_roll_padd_lo_conv_ds[:, filt_len//2:, :, :]], axis=1)

    # crop to needed dims
    x_norm_roll_padd_lo_conv_ds_fix_crop = x_norm_roll_padd_lo_conv_ds_fix[:, 0:h//2, :, :]

    # output
    with tf.Session() as sess:
        test = sess.run(x_norm_roll_padd_lo_conv_ds_fix_crop)

    write_raw.tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\python_roll_padd_conv_last.hex",
                                  test[0, ..., 0])
    return


Faf, Fsf = filters.FSfarras()
af, sf = filters.duelfilt()

# [cA, cH, cV, cD] =
analysis_filter_bank2d(img_grey, Faf[0][0], Faf[0][1], Faf[1][0], Faf[1][1])

pass