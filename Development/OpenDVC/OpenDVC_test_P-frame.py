import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_compression as tfc
from scipy import misc
import motion
import os
import imageio
import cv2
# tf.executing_eagerly()
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

# Y0_com_img_tf = tf.convert_to_tensor(Y0_com_img, dtype=tf.float32)

# flow_tensor = motion.optical_flow(Y0_com_img_tf, Y0_com_img_tf, 1, h, w)

# motion estimation
x_inp1 = layers.Input(shape=(h, w, c))
x_inp2 = layers.Input(shape=(h, w, c))
x = motion.OpticalFlow()([x_inp1, x_inp2])
vt = Model(inputs=[x_inp1, x_inp2], outputs=x, name="MyModel")
aout = vt.predict([Y0_com_img, Y0_com_img])
# model.summary()

# motion vector out

print(aout.max(), aout.min())
