
import numpy as np
import math
import tensorflow as tf


def tf_to_ndarray(data, datatype=tf.float64):
    data = tf.image.convert_image_dtype(data[0, ..., 0], dtype=datatype)
    with tf.Session() as sess:
        data = sess.run(data)
    return data


def cast_like_matlab_uint8_2d(data):

    h, w = data.shape
    for row in range(h):
        for col in range(w):
            frac, integ = math.modf(data[row,col])
            if frac > 0.5:
                data[row, col] = np.ceil(data[row, col])
            elif frac <= 0.5:
                data[row, col] = np.floor(data[row, col])

    data_clip = np.clip(data, 0, 255)
    return data_clip.astype('uint8')