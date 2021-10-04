import math
import numpy as np
import tensorflow as tf
from tensorflow_wavelets.utils.cast import cast_like_matlab_uint8_2d


def write_raw(file_path, data):
    with open(file_path, "wb") as outfile:
        outfile.write(data)  # Write it


def tensor_to_write_raw(file_path, tensor_data, dastype='uint8'):
    out_img = tf.image.convert_image_dtype(tensor_data, dtype=tf.float32)
    with tf.Session() as sess:
        out_img = sess.run(out_img)
    out_img = cast_like_matlab_uint8_2d(out_img)
    write_raw(file_path, out_img)
