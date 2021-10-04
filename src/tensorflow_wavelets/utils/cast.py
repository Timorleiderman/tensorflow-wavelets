
import numpy as np
import math
import tensorflow as tf


def tf_1d_to_ndarray(data, datatype=tf.float64):

    with tf.Session() as sess:
        data = sess.run(data)
    return data


def tf_to_ndarray(data, datatype=tf.float32):
    data = tf.image.convert_image_dtype(data[0, ..., 0], dtype=datatype)
    with tf.Session() as sess:
        data = sess.run(data)
    return data


def tf_rgb_to_ndarray(data, datatype=tf.float32):
    data = tf.image.convert_image_dtype(data[0, ..., :], dtype=datatype)
    with tf.Session() as sess:
        data = sess.run(data)
    return data


def tf2_rgb_to_ndarray(data, datatype=tf.float32):
    data = tf.image.convert_image_dtype(data[0, ..., :], dtype=datatype)
    return data


def tf_rank4_to_ndarray(data, datatype=tf.float32):
    data = tf.image.convert_image_dtype(data[0, ..., 0], dtype=datatype)
    with tf.Session() as sess:
        data = sess.run(data)
    return data


def tf_rank2_to_ndarray(data, datatype=tf.float32):
    data = tf.image.convert_image_dtype(data, dtype=datatype)
    with tf.Session() as sess:
        data = sess.run(data)
    return data


def cast_like_matlab_uint8_2d_rgb(data):
    data = np.clip(data, 0, 255)
    h, w, c = data.shape
    for ch in range(c):
        for row in range(h):
            for col in range(w):
                frac, integ = math.modf(data[row, col, ch])
                if frac > 0.5:
                    data[row, col, ch] = np.ceil(data[row, col, ch])
                elif frac <= 0.5:
                    data[row, col, ch] = np.floor(data[row, col, ch])

    return data.astype('uint8')


def cast_like_matlab_uint8_2d(data):
    data = np.clip(data, 0, 255)
    h, w = data.shape
    for row in range(h):
        for col in range(w):
            frac, integ = math.modf(data[row,col])
            if frac > 0.5:
                data[row, col] = np.ceil(data[row, col])
            elif frac <= 0.5:
                data[row, col] = np.floor(data[row, col])


    return data.astype('uint8')