import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import filters

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

img_in = cv2.imread("../input/Lenna_Orig.png")
img_grey = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
w, h = img_grey.shape


def analysis_filter_bank2d(x, lod_row, hid_row, lod_col, hid_col):

    conv_type = 'same'
    h, w = x.shape
    filt_len = len(lod_row)
    x = tf.roll(x, shift=filt_len, axis=-1)

    x = tf.image.convert_image_dtype(x, dtype=tf.float32)

    with tf.Session() as sess:
        x = sess.run(x)
    cv2.imshow("lenna", x)
    cv2.waitKey(0)
    print(filt_len)
    return


Faf, Fsf = filters.FSfarras()
af, sf = filters.duelfilt()

# [cA, cH, cV, cD] =
analysis_filter_bank2d(img_grey, Faf[0][0], Faf[0][1], Faf[1][0], Faf[1][1])

pass