import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.python.keras.engine import training
import tensorflow_compression as tfc
import tensorflow_addons as tfa
from scipy import misc
import motion
import os
import imageio
import cv2
tf.executing_eagerly()
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


batch_size = 1
Channel = 3

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ref", default='ref.png')
parser.add_argument("--raw", default='raw.png')
parser.add_argument("--com", default='com.png')
parser.add_argument("--bin", default='bitstream.bin')
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--metric", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])

args = parser.parse_args()

Y0_com_img = cv2.imread(args.ref) / 255.0
Y1_com_img = cv2.imread("/workspaces/tensorflow-wavelets/Development/OpenDVC/BasketballPass/f055.png") / 255.0

Y1_raw_img = cv2.imread(args.raw)

Y0_com_img = np.expand_dims(Y0_com_img, 0)
Y1_com_img = np.expand_dims(Y1_com_img, 0)

Y1_raw_img = np.expand_dims(Y1_raw_img, 0)

h = Y0_com_img.shape[1]
w = Y0_com_img.shape[2]
c = Y0_com_img.shape[3]
batch_size = 4
lr_init = 1e-4
l = 256

folder = np.load("/workspaces/tensorflow-wavelets/Development/OpenDVC/vimeo_npy.py")

Y0_com_img_tf = tf.convert_to_tensor(Y0_com_img, dtype=tf.float32)
Y1_raw_img_tf = tf.convert_to_tensor(Y1_raw_img, dtype=tf.float32)


# imgs input (I frame and P frame) shape for example (1, 240, 416, 3) 
# optical flow -> CNN model to estimate motion information "vt" 
vt, loss_0, loss_1, loss_2, loss_3, loss_4 = motion.optical_flow(Y0_com_img_tf, Y1_raw_img_tf, 1, h, w)
# (1, 240, 416, 2) -> x,y for motion vectors 

# MV encoder input from optical flow
mt = motion.encoder(vt, num_filters=128, kernel_size=3, M=128)
# (1, 15, 26, 128)

# Entropy bottelneck
noisy = tfc.NoisyNormal(loc=.5, scale=8.)
entropy_quantizer_mv = tfc.ContinuousBatchedEntropyModel(noisy, 1, compression=True)
string_mv = entropy_quantizer_mv.compress(mt)
# (1, 15, 16)
string_mv = tf.squeeze(string_mv, axis=0)
#  (15, 16)

# mt_hat and mv_likelihood is like bits
mt_hat, MV_likelihoods = entropy_quantizer_mv(mt, training=True)
#(1,15,26,128), (1, 15, 26)

vt_hat = motion.decoder(mt_hat, num_filters=128, kernel_size=3, M=2)
# (1, 240, 416, 2)

Y1_warp = tfa.image.dense_image_warp(Y0_com_img_tf, vt_hat )
# (1, 240, 416, 3)

# motion compenstation
MC_input = tf.concat([vt_hat, Y0_com_img_tf, Y1_warp], axis=-1)
# shape (1, 240 416, 8)
Y1_MC = motion.MotionCompensation(MC_input)

Res = Y1_raw_img_tf - Y1_MC
# (1, 240, 416, 3)

res_latent = motion.encoder(Res, num_filters=128, kernel_size=5, M=128)

entropy_quantizer_res = tfc.ContinuousBatchedEntropyModel(noisy, 1, compression=True)
string_res = entropy_quantizer_res.compress(res_latent)
string_res = tf.squeeze(string_res, axis=0)
res_latent_hat, Res_likelihoods = entropy_quantizer_res(res_latent)
Res_hat = motion.decoder(res_latent_hat, num_filters=128, kernel_size=5, M=3)

# Reconstructed frame
Y1_com = Res_hat + Y1_MC

# Total number of bits divided by number of pixels.
train_bpp_MV = tf.math.reduce_sum(tf.math.log(MV_likelihoods)) / (-np.log(2) * h * w * batch_size)
train_bpp_Res = tf.math.reduce_sum(tf.math.log(Res_likelihoods)) / (-np.log(2) * h * w * batch_size)

train_mse = tf.math.reduce_mean(tf.math.squared_difference(Y1_com, Y1_raw_img_tf))
quality = 10.0*tf.math.log(1.0/train_mse)/tf.math.log(10.0)

# Mean squared error across pixels.
total_mse = tf.math.reduce_mean(tf.math.squared_difference(Y1_com, Y1_raw_img_tf))
warp_mse = tf.math.reduce_mean(tf.math.squared_difference(Y1_warp, Y1_raw_img_tf))
MC_mse = tf.math.reduce_mean(tf.math.squared_difference(Y1_raw_img_tf, Y1_MC))

psnr = 10.0*tf.math.log(1.0/total_mse)/tf.math.log(10.0)

train_loss_total = l * total_mse + (train_bpp_MV + train_bpp_Res)
train_loss_MV = l * warp_mse + train_bpp_MV
train_loss_MC = l * MC_mse + train_bpp_MV


train_MV = tf.optimizers.Adam(learning_rate=lr_init) # .minimize(train_loss_MV)
train_MC = tf.optimizers.Adam(learning_rate=lr_init) #.minimize(train_loss_MC)
train_total = tf.optimizers.Adam(learning_rate=lr_init) #.minimize(train_loss_total)


@tf.function
def train_step(images):
      pass
# data = tf.image.convert_image_dtype(Y1_com[0, ..., :], dtype=tf.uint8)
# cv2.imwrite("/workspaces/tensorflow-wavelets/Development/OpenDVC/BasketballPass_com/data_out.png", data.numpy())
# bpp = (2 + len(string_mv) + len(string_res)) * 8 / h / w

# print(args.metric + ' = ' + str(quality), 'bpp = ' + str(bpp))
# motion estimation
# x_inp1 = layers.Input(shape=(h, w, c))
# x_inp2 = layers.Input(shape=(h, w, c))
# x = motion.OpticalFlow()([x_inp1, x_inp2])
# vt = Model(inputs=[x_inp1, x_inp2], outputs=x, name="MyModel")
# aout = vt.predict([Y0_com_img, Y0_com_img])
# # model.summary()
# # motion vector out
# print(aout.max(), aout.min())

# modtion vector encoder net

