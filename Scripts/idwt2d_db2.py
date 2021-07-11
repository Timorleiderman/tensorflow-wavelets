import cv2
import math
import pywt
import numpy as np
import models.DWT2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# load Lenna image
img_in = cv2.imread("../input/Lenna_Orig.png")
img_grey = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
w, h = img_grey.shape

# load DWT model
model = keras.Sequential()
model.add(keras.Input(shape=(512, 512, 1)))
model.add(models.DWT2.DWT())
model.summary()

# convert input image to tensor in order to pass to model
img_grey = tf.expand_dims(img_grey, axis=0)
img_grey = tf.expand_dims(img_grey, axis=-1)
coeffs = model.predict(img_grey, steps=1)

# convert model output to images
# cA = tf.image.convert_image_dtype(coeffs[0, ..., 0], dtype=tf.float32)
# cH = tf.image.convert_image_dtype(coeffs[0, ..., 1], dtype=tf.float32)
# cV = tf.image.convert_image_dtype(coeffs[0, ..., 2], dtype=tf.float32)
# cD = tf.image.convert_image_dtype(coeffs[0, ..., 3], dtype=tf.float32)
#
# with tf.Session() as sess:
#     cA = sess.run(cA)
#     cH = sess.run(cH)
#     cV = sess.run(cV)
#     cD = sess.run(cD)


# 3rd party library to compare resaoults
# coeffs = pywt.dwt2(img_grey, 'db2')
# cA, (cH, cV, cD) = coeffs


# convert coeffs into tensor for IDWT model
# cA = tf.constant(cA)
# cA = tf.expand_dims(cA, axis=-1)
# cH = tf.constant(cH)
# cH = tf.expand_dims(cH, axis=-1)
# cV = tf.constant(cV)
# cV = tf.expand_dims(cV, axis=-1)
# cD = tf.constant(cD)
# cD = tf.expand_dims(cD, axis=-1)
# x = tf.concat([cA, cH, cV, cD], axis=-1)
# x = tf.expand_dims(x, axis=0)



# symetric border padding colums for convolution
x = tf.pad(coeffs, [[0, 0], [0, 0], [3, 3], [0, 0]], "SYMMETRIC")
x = tf.cast(x, tf.float32)

# calc db2 coefs
db2_h0 = (1+math.sqrt(3))/(4*math.sqrt(2))
db2_h1 = (3+math.sqrt(3))/(4*math.sqrt(2))
db2_h2 = (3-math.sqrt(3))/(4*math.sqrt(2))
db2_h3 = (1-math.sqrt(3))/(4*math.sqrt(2))

# Reconstruction LPF and HPF
db2_lpfR = [db2_h3, db2_h1, db2_h2, db2_h0]
db2_hpfR = [-db2_h0, db2_h1, -db2_h2, db2_h3]

# convert to matrix for conv2d
db2_lpf = tf.constant(db2_lpfR)
db2_lpf = tf.reshape(db2_lpf, (1, 4, 1, 1))
db2_lpf = tf.repeat(db2_lpf, 4, axis=-1)

db2_hpf = tf.constant(db2_hpfR)
db2_hpf = tf.reshape(db2_hpf, (1, 4, 1, 1))
db2_hpf = tf.repeat(db2_hpf, 4, axis=-1)

# upsampling -> padding zeros between all elements

zero_tensor = tf.zeros(shape=x.shape, dtype=tf.float32)
c = tf.stack([x, zero_tensor], axis=4)
a_us = tf.reshape(c, shape=[x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]])

zero_tensor_n = tf.zeros(shape=a_us.shape, dtype=tf.float32)

d = tf.stack([a_us, zero_tensor_n], axis=3)
a_us_us = tf.reshape(d, shape=[x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]])


LL_us_pad = tf.expand_dims(a_us_us[:,:,:,0], axis=-1)
LH_us_pad = tf.expand_dims(a_us_us[:,:,:,1], axis=-1)
HL_us_pad = tf.expand_dims(a_us_us[:,:,:,2], axis=-1)
HH_us_pad = tf.expand_dims(a_us_us[:,:,:,3], axis=-1)

LL_conv_lpf = tf.nn.conv2d(LL_us_pad, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],)
LL_conv_lpf = tf.transpose(LL_conv_lpf, perm=[0, 2, 1,3])
LL_conv_lpf_lpf = tf.nn.conv2d(LL_conv_lpf, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],)

LH_conv_lpf = tf.nn.conv2d(LH_us_pad, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],)
LH_conv_lpf = tf.transpose(LH_conv_lpf, perm=[0, 2, 1,3])
LH_conv_lpf_hpf = tf.nn.conv2d(LH_conv_lpf, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],)

HL_conv_hpf = tf.nn.conv2d(HL_us_pad, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],)
HL_conv_hpf = tf.transpose(HL_conv_hpf, perm=[0, 2, 1,3])
HL_conv_hpf_lpf = tf.nn.conv2d(HL_conv_hpf, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],)

HH_conv_hpf = tf.nn.conv2d(HH_us_pad, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],)
HH_conv_hpf = tf.transpose(HH_conv_hpf, perm=[0, 2, 1,3])
HH_conv_hpf_hpf = tf.nn.conv2d(HH_conv_hpf, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],)

LL_LH = tf.add(LL_conv_lpf_lpf, LH_conv_lpf_hpf)
HL_HH = tf.add(HL_conv_hpf_lpf, HH_conv_hpf_hpf)

img = tf.add(LL_LH,HL_HH)


















#
# with tf.Session() as sess:
#     # cA_Temp = sess.run(cA_Temp)
#     a_us_Temp = sess.run(conv_rows_lpf)
#     # a_us_TEMP = sess.run(a_us_assign)
#     pass
# #

# conv_rows_hpf = tf.nn.conv2d(
#     x, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
# )
#
# conv_rows_lpf_ds = conv_rows_lpf[:, :, 1:w:2, :]
# conv_rows_hpf_ds = conv_rows_hpf[:, :, 1:w:2, :]
#
#
# conv_rows_lpf_ds_padd = tf.pad(conv_rows_lpf_ds, [[0, 0], [3, 3], [0, 0], [0, 0]], "SYMMETRIC")
# conv_rows_hpf_ds_padd = tf.pad(conv_rows_lpf_ds, [[0, 0], [3, 3], [0, 0], [0, 0]], "SYMMETRIC")
#
# conv_rows_lpf_ds_padd = tf.transpose(conv_rows_lpf_ds_padd)
# conv_rows_hpf_ds_padd = tf.transpose(conv_rows_hpf_ds_padd)
#
# conv_rows_lps_padd_conv_cols_lpf = tf.nn.conv2d(
#     conv_rows_lpf_ds_padd, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],
# )
# conv_rows_lps_padd_conv_cols_hpf = tf.nn.conv2d(
#     conv_rows_lpf_ds_padd, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
# )
# conv_rows_hps_padd_conv_cols_lpf = tf.nn.conv2d(
#     conv_rows_hpf_ds_padd, db2_lpf, padding='VALID', strides=[1, 1, 1, 1],
# )
# conv_rows_hps_padd_conv_cols_hpf = tf.nn.conv2d(
#     conv_rows_hpf_ds_padd, db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
# )
#
# conv_rows_lps_padd_conv_cols_lpf = tf.transpose(conv_rows_lps_padd_conv_cols_lpf)
# conv_rows_lps_padd_conv_cols_hpf = tf.transpose(conv_rows_lps_padd_conv_cols_hpf)
# conv_rows_hps_padd_conv_cols_lpf = tf.transpose(conv_rows_hps_padd_conv_cols_lpf)
# conv_rows_hps_padd_conv_cols_hpf = tf.transpose(conv_rows_hps_padd_conv_cols_hpf)
#
# conv_rows_lps_padd_conv_cols_lpf_ds = conv_rows_lps_padd_conv_cols_lpf[:, 1:h:2, :, :]
# conv_rows_lps_padd_conv_cols_hpf_ds = conv_rows_lps_padd_conv_cols_hpf[:, 1:h:2, :, :]
# conv_rows_hps_padd_conv_cols_lpf_ds = conv_rows_hps_padd_conv_cols_lpf[:, 1:h:2, :, :]
# conv_rows_hps_padd_conv_cols_hpf_ds = conv_rows_hps_padd_conv_cols_hpf[:, 1:h:2, :, :]
#
# image_tensor_conv_rows_lpf = tf.image.convert_image_dtype(conv_rows_lpf[0, ..., 0], dtype=tf.float32)
# image_tensor_conv_rows_hpf = tf.image.convert_image_dtype(conv_rows_hpf[0, ..., 0], dtype=tf.float32)
#
# image_tensor_conv_rows_lpf_ds = tf.image.convert_image_dtype(conv_rows_lpf_ds[0, ..., 0], dtype=tf.float32)
# image_tensor_conv_rows_hpf_ds = tf.image.convert_image_dtype(conv_rows_hpf_ds[0, ..., 0], dtype=tf.float32)
#
# img_conv_rows_lps_padd_conv_cols_lpf_ds = tf.image.convert_image_dtype(conv_rows_lps_padd_conv_cols_lpf_ds[0, ..., 0], dtype=tf.float32)
# img_conv_rows_lps_padd_conv_cols_hpf_ds = tf.image.convert_image_dtype(conv_rows_lps_padd_conv_cols_hpf_ds[0, ..., 0], dtype=tf.float32)
# img_conv_rows_hps_padd_conv_cols_lpf_ds = tf.image.convert_image_dtype(conv_rows_hps_padd_conv_cols_lpf_ds[0, ..., 0], dtype=tf.float32)
# img_conv_rows_hps_padd_conv_cols_hpf_ds = tf.image.convert_image_dtype(conv_rows_hps_padd_conv_cols_hpf_ds[0, ..., 0], dtype=tf.float32)
#
#
# orig_iamge_padded = tf.image.convert_image_dtype(x[0, ..., 0], dtype=tf.uint8)
# with tf.Session() as sess:
#     # print(sess.run(db2_lpf))
#     # print(sess.run(db2_hpf))
#     image_conv_rows_lpf = sess.run(image_tensor_conv_rows_lpf)
#     image_conv_rows_hpf = sess.run(image_tensor_conv_rows_hpf)
#     image_conv_rows_lpf_ds = sess.run(image_tensor_conv_rows_lpf_ds)
#     image_conv_rows_hpf_ds = sess.run(image_tensor_conv_rows_hpf_ds)
#
#     LL = sess.run(img_conv_rows_lps_padd_conv_cols_lpf_ds)
#     LH = sess.run(img_conv_rows_lps_padd_conv_cols_hpf_ds)
#     HL = sess.run(img_conv_rows_hps_padd_conv_cols_lpf_ds)
#     HH = sess.run(img_conv_rows_hps_padd_conv_cols_hpf_ds)
#     orig_img_pad = sess.run(orig_iamge_padded)
#     pass

# cv2.imshow("tf", image)
# cv2.waitKey(0)
# LL = np.clip(LL,0,255)
# LL = np.ceil(LL)
# LL = LL.astype("uint8")
# with open(r"D:\TEMP\LL_python.raw", "wb") as outfile:
#     outfile.write(LL)  # Write it
