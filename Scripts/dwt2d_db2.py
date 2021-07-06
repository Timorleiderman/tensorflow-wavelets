import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def kernelInitializerSobelx(shape, dtype=None):
    sobel_x = tf.constant(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype="float32"
    )
    #create the missing dims.
    print(sobel_x.shape)
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))
    print(sobel_x.shape)
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2], shape[-1]))
    print(sobel_x.shape)
    print("sobel end")
    return sobel_x


img_in = cv2.imread("../input/Lenna_Orig.png")
img_grey = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
# x = cv2.transpose(x)
# expend dimps to look like x_train
x = tf.expand_dims(img_grey, axis=0)
x = tf.expand_dims(x, axis=-1)

# symetric border padding colums for convolution
x = tf.pad(x, [[0, 0], [0, 0], [3, 3], [0, 0]], "SYMMETRIC")
x = tf.cast(x, tf.float32)/255.0

# filter = kernelInitializerLoDdb2((4, 4, 1, 1))

# ans = tf.nn.conv2d((b/255.0),
#                    filter,
#                    strides=[1, 1, 1, 1],
#                    padding='VALID')
# calc db2 coefs
db2_h0 = (1+math.sqrt(3))/(4*math.sqrt(2))
db2_h1 = (3+math.sqrt(3))/(4*math.sqrt(2))
db2_h2 = (3-math.sqrt(3))/(4*math.sqrt(2))
db2_h3 = (1-math.sqrt(3))/(4*math.sqrt(2))
db2_lpf = [db2_h0, db2_h1, db2_h2, db2_h3]

# convert to matrix for conv2d
db2_hpf = [db2_h3, -db2_h2, db2_h1, -db2_h0]
db2_hpf = tf.constant(db2_hpf)
db2_hpf = tf.reshape(db2_hpf, (1, 4, 1, 1))

db2_lpf = tf.constant(db2_lpf)
db2_lpf = tf.reshape(db2_lpf, (1, 4, 1, 1))

conv_rows_lpf = tf.nn.conv2d(
    x, db2_lpf, padding='VALID',
)
conv_rows_hpf = tf.nn.conv2d(
    x, db2_hpf, padding='VALID',
)

conv_rows_lpf_ds = conv_rows_lpf[:, 0::2, 0::2, :]
conv_rows_hpf_ds = conv_rows_hpf[:, 0::2, 0::2, :]
# print(ans.shape)
# with tf.Session() as sess:
#      print(sess.run(ans))
#      # print( sess.run( db2_lpf ) )
#
image_tensor_conv_rows_lpf = tf.image.convert_image_dtype(conv_rows_lpf[0, ..., 0], dtype=tf.float32)*255
image_tensor_conv_rows_hpf = tf.image.convert_image_dtype(conv_rows_hpf[0, ..., 0], dtype=tf.float32)*255
orig_iamge_padded = tf.image.convert_image_dtype(x[0, ..., 0], dtype=tf.uint8)
with tf.Session() as sess:
    image_conv_rows_lpf = sess.run(image_tensor_conv_rows_lpf)
    image_conv_rows_hpf = sess.run(image_tensor_conv_rows_hpf)
    o_img_pad = sess.run(orig_iamge_padded)
    print(sess.run(db2_lpf))
    print(sess.run(db2_hpf))

# cv2.imshow("tf", image)
# cv2.waitKey(0)