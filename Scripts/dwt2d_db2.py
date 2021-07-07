import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_in = cv2.imread("../input/Lenna_Orig.png")
img_grey = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
w, h = img_grey.shape


# expend dimps to look like x_train
x = tf.expand_dims(img_grey, axis=0)
x = tf.expand_dims(x, axis=-1)

# symetric border padding colums for convolution
x = tf.pad(x, [[0, 0], [0, 0], [3, 3], [0, 0]], "SYMMETRIC")
x = tf.cast(x, tf.float32)

# calc db2 coefs
db2_h0 = (1+math.sqrt(3))/(4*math.sqrt(2))
db2_h1 = (3+math.sqrt(3))/(4*math.sqrt(2))
db2_h2 = (3-math.sqrt(3))/(4*math.sqrt(2))
db2_h3 = (1-math.sqrt(3))/(4*math.sqrt(2))

db2_lpf = [db2_h0, db2_h1, db2_h2, db2_h3]
db2_hpf = [db2_h3, -db2_h2, db2_h1, -db2_h0]

# convert to matrix for conv2d
db2_lpf = tf.constant(db2_lpf)
db2_lpf = tf.reshape(db2_lpf, (1, 4, 1, 1))

db2_hpf = tf.constant(db2_hpf)
db2_hpf = tf.reshape(db2_hpf, (1, 4, 1, 1))

conv_rows_lpf = tf.nn.conv2d(
    x, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],
)
conv_rows_hpf = tf.nn.conv2d(
    x, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
)

conv_rows_lpf_ds = conv_rows_lpf[:, :, 1:w:2, :]
conv_rows_hpf_ds = conv_rows_hpf[:, :, 1:w:2, :]


conv_rows_lpf_ds_padd = tf.pad(conv_rows_lpf_ds, [[0, 0], [3, 3], [0, 0], [0, 0]], "SYMMETRIC")
conv_rows_hpf_ds_padd = tf.pad(conv_rows_lpf_ds, [[0, 0], [3, 3], [0, 0], [0, 0]], "SYMMETRIC")

conv_rows_lpf_ds_padd = tf.transpose(conv_rows_lpf_ds_padd)
conv_rows_hpf_ds_padd = tf.transpose(conv_rows_hpf_ds_padd)

conv_rows_lps_padd_conv_cols_lpf = tf.nn.conv2d(
    conv_rows_lpf_ds_padd, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],
)
conv_rows_lps_padd_conv_cols_hpf = tf.nn.conv2d(
    conv_rows_lpf_ds_padd, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
)
conv_rows_hps_padd_conv_cols_lpf = tf.nn.conv2d(
    conv_rows_hpf_ds_padd, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],
)
conv_rows_hps_padd_conv_cols_hpf = tf.nn.conv2d(
    conv_rows_hpf_ds_padd, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
)

conv_rows_lps_padd_conv_cols_lpf = tf.transpose(conv_rows_lps_padd_conv_cols_lpf)
conv_rows_lps_padd_conv_cols_hpf = tf.transpose(conv_rows_lps_padd_conv_cols_hpf)
conv_rows_hps_padd_conv_cols_lpf = tf.transpose(conv_rows_hps_padd_conv_cols_lpf)
conv_rows_hps_padd_conv_cols_hpf = tf.transpose(conv_rows_hps_padd_conv_cols_hpf)

conv_rows_lps_padd_conv_cols_lpf_ds = conv_rows_lps_padd_conv_cols_lpf[:, 1:h:2, :, :]
conv_rows_lps_padd_conv_cols_hpf_ds = conv_rows_lps_padd_conv_cols_hpf[:, 1:h:2, :, :]
conv_rows_hps_padd_conv_cols_lpf_ds = conv_rows_hps_padd_conv_cols_lpf[:, 1:h:2, :, :]
conv_rows_hps_padd_conv_cols_hpf_ds = conv_rows_hps_padd_conv_cols_hpf[:, 1:h:2, :, :]

image_tensor_conv_rows_lpf = tf.image.convert_image_dtype(conv_rows_lpf[0, ..., 0], dtype=tf.float32)
image_tensor_conv_rows_hpf = tf.image.convert_image_dtype(conv_rows_hpf[0, ..., 0], dtype=tf.float32)

image_tensor_conv_rows_lpf_ds = tf.image.convert_image_dtype(conv_rows_lpf_ds[0, ..., 0], dtype=tf.float32)
image_tensor_conv_rows_hpf_ds = tf.image.convert_image_dtype(conv_rows_hpf_ds[0, ..., 0], dtype=tf.float32)

img_conv_rows_lps_padd_conv_cols_lpf_ds = tf.image.convert_image_dtype(conv_rows_lps_padd_conv_cols_lpf_ds[0, ..., 0], dtype=tf.float32)
img_conv_rows_lps_padd_conv_cols_hpf_ds = tf.image.convert_image_dtype(conv_rows_lps_padd_conv_cols_hpf_ds[0, ..., 0], dtype=tf.float32)
img_conv_rows_hps_padd_conv_cols_lpf_ds = tf.image.convert_image_dtype(conv_rows_hps_padd_conv_cols_lpf_ds[0, ..., 0], dtype=tf.float32)
img_conv_rows_hps_padd_conv_cols_hpf_ds = tf.image.convert_image_dtype(conv_rows_hps_padd_conv_cols_hpf_ds[0, ..., 0], dtype=tf.float32)


orig_iamge_padded = tf.image.convert_image_dtype(x[0, ..., 0], dtype=tf.uint8)
with tf.Session() as sess:
    # print(sess.run(db2_lpf))
    # print(sess.run(db2_hpf))
    image_conv_rows_lpf = sess.run(image_tensor_conv_rows_lpf)
    image_conv_rows_hpf = sess.run(image_tensor_conv_rows_hpf)
    image_conv_rows_lpf_ds = sess.run(image_tensor_conv_rows_lpf_ds)
    image_conv_rows_hpf_ds = sess.run(image_tensor_conv_rows_hpf_ds)

    LL = sess.run(img_conv_rows_lps_padd_conv_cols_lpf_ds)
    LH = sess.run(img_conv_rows_lps_padd_conv_cols_hpf_ds)
    HL = sess.run(img_conv_rows_hps_padd_conv_cols_lpf_ds)
    HH = sess.run(img_conv_rows_hps_padd_conv_cols_hpf_ds)
    orig_img_pad = sess.run(orig_iamge_padded)
    pass

# cv2.imshow("tf", image)
# cv2.waitKey(0)
# LL = np.clip(LL,0,255)
# LL = np.ceil(LL)
# LL = LL.astype("uint8")
# with open(r"D:\TEMP\LL_python.raw", "wb") as outfile:
#     outfile.write(LL)  # Write it
