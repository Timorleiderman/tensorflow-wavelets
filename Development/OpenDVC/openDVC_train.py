import os
import cv2
import motion
import imageio
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_compression as tfc


from scipy import misc
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.python.keras.engine import training


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

print(args)

batch_size = 4
Height = 256
Width = 256
Channel = 3
lr_init = 1e-4
frames=2
I_QP = 27
l = 256


inputs_y0_com = tf.keras.Input(shape=(Width, Height, Channel,))
inputs_y1_raw = tf.keras.Input(shape=(Width, Height, Channel,))


# imgs input (I frame and P frame) shape for example (1, 240, 416, 3) 
# optical flow -> CNN model to estimate motion information "vt" 
vt, loss_0, loss_1, loss_2, loss_3, loss_4 = motion.optical_flow(inputs_y0_com, inputs_y1_raw, batch_size, Height, Width)
# (1, 240, 416, 2) -> x,y for motion vectors 

# MV encoder input from optical flow
mt = motion.encoder(vt, num_filters=128, kernel_size=3, M=128)
# (1, 15, 26, 128)


# model = tf.keras.Model([inputs_y0_com, inputs_y1_raw], mt, name="mymodel")


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

Y1_warp = tfa.image.dense_image_warp(inputs_y0_com, vt_hat )
# (1, 240, 416, 3)

# motion compenstation
MC_input = tf.concat([vt_hat, inputs_y0_com, Y1_warp], axis=-1)
# shape (1, 240 416, 8)
Y1_MC = motion.MotionCompensation(MC_input)

Res = inputs_y1_raw - Y1_MC
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
train_bpp_MV = tf.math.reduce_sum(tf.math.log(MV_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
train_bpp_Res = tf.math.reduce_sum(tf.math.log(Res_likelihoods)) / (-np.log(2) * Height * Width * batch_size)

train_mse = tf.math.reduce_mean(tf.math.squared_difference(Y1_com, inputs_y1_raw))
quality = 10.0*tf.math.log(1.0/train_mse)/tf.math.log(10.0)

# Mean squared error across pixels.
total_mse = tf.math.reduce_mean(tf.math.squared_difference(Y1_com, inputs_y1_raw))
warp_mse = tf.math.reduce_mean(tf.math.squared_difference(Y1_warp, inputs_y1_raw))
MC_mse = tf.math.reduce_mean(tf.math.squared_difference(inputs_y1_raw, Y1_MC))

psnr = 10.0*tf.math.log(1.0/total_mse)/tf.math.log(10.0)

train_loss_total = l * total_mse + (train_bpp_MV + train_bpp_Res)
train_loss_MV = l * warp_mse + train_bpp_MV
train_loss_MC = l * MC_mse + train_bpp_MV


train_MV = tf.optimizers.Adam(learning_rate=lr_init) # .minimize(train_loss_MV)
train_MC = tf.optimizers.Adam(learning_rate=lr_init) #.minimize(train_loss_MC)
train_total = tf.optimizers.Adam(learning_rate=lr_init) #.minimize(train_loss_total)


