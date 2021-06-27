import os
import cv2
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from skimage import data
import matplotlib.pyplot as plt

# load camera man
img = np.expand_dims(data.camera(), -1)
img = np.expand_dims(img, 0)  # shape: (1, 512, 512, 1)

def kernelInitializerSobelx(shape, dtype=None):
    sobel_x = tf.constant(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype="float32"
    )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2], shape[-1]))
    return sobel_x

def kernelInitializerSobely(shape, dtype=None):
    sobel_x = tf.constant(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype="float32"
    )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2], shape[-1]))
    return sobel_x


sobel_x = np.array([[-0.25, -0.2 ,  0.  ,  0.2 ,  0.25],
                    [-0.4 , -0.5 ,  0.  ,  0.5 ,  0.4 ],
                    [-0.5 , -1.  ,  0.  ,  1.  ,  0.5 ],
                    [-0.4 , -0.5 ,  0.  ,  0.5 ,  0.4 ],
                    [-0.25, -0.2 ,  0.  ,  0.2 ,  0.25]])

sobel_y = np.array([[-0.25, -0.4 , -0.5 , -0.4 , -0.25],
                    [-0.2 , -0.5 , -1.  , -0.5 , -0.2 ],
                    [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
                    [ 0.2 ,  0.5 ,  1.  ,  0.5 ,  0.2 ],
                    [ 0.25,  0.4 ,  0.5 ,  0.4 ,  0.25]])

filters = np.concatenate([[sobel_x], [sobel_y]])  # shape: (2, 5, 5)
filters = np.expand_dims(filters, -1)  # shape: (2, 5, 5, 1)
filters = filters.transpose(1, 2, 3, 0)  # shape: (5, 5, 1, 2)

# Convolve image
filterx = kernelInitializerSobelx((3, 3, 1, 1))
filtery = kernelInitializerSobely((3, 3, 1, 1))

print(filterx.shape)

ans = tf.nn.conv2d((img / 255.0).astype('float32'),
                   filters,
                   strides=[1, 1, 1, 1],
                   padding='SAME')


ansx = tf.nn.conv2d((img / 255.0).astype('float32'),
                   filterx,
                   strides=[1, 1, 1, 1],
                   padding='SAME')

ansy = tf.nn.conv2d((img / 255.0).astype('float32'),
                    filtery,
                    strides=[1, 1, 1, 1],
                    padding='SAME')

with tf.Session() as sess:
    ans_np = sess.run(ans)  # shape: (1, 512, 512, 2)

    ans_npx = sess.run(ansx)  # shape: (1, 512, 512, 1)
    ans_npy = sess.run(ansy)  # shape: (1, 512, 512, 1)

filtered1 = ans_np[0, ..., 0]
filtered2 = ans_np[0, ..., 1]

filtered3 = ans_npx[0, ..., 0]
filtered4 = ans_npy[0, ..., 0]

f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)

axarr[0].imshow((filtered1).astype('uint8'), interpolation='none', cmap='gray', vmin=0, vmax=255)
axarr[1].imshow((filtered2).astype('uint8'), interpolation='none', cmap='gray', vmin=0, vmax=255)

axarr[2].imshow((filtered3).astype('uint8'), interpolation='none', cmap='gray', vmin=0, vmax=255)
axarr[3].imshow((filtered4).astype('uint8'), interpolation='none', cmap='gray', vmin=0, vmax=255)
axarr[4].imshow((img[0]).astype('uint8'), interpolation='none', cmap='gray', vmin=0, vmax=255)

plt.show()