import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Conv2D
import tensorflow_addons as tfa
from tensorflow.keras import layers
import tensorflow_compression as tfc
import numpy as np


def convnet(im1_warp, im2, flow):

    
    input = tf.concat([im1_warp, im2, flow], axis=-1)

    conv1 = Conv2D(filters=32, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(input)
    conv2 = Conv2D(filters=64, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(conv1)
    conv3 = Conv2D(filters=32, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(conv2)
    conv4 = Conv2D(filters=16, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(conv3)
    conv5 = Conv2D(filters=2 , kernel_size=[7, 7], padding="same", activation=None)(conv4)

    return conv5


def loss(flow_course, im1, im2):

    # flow_shape = [-1, im1.shape[1], im1.shape[2], 2]

    flow = tf.image.resize(flow_course, [tf.shape(im1)[1], tf.shape(im2)[2]])
    # flow = tf.image.resize_images(flow_course, [tf.shape(im1)[1], tf.shape(im2)[2]])
    # im1_warped = tfa.image.dense_image_warp(im1, flow )
    im1_warped = tf.keras.layers.Lambda(lambda a: tfa.image.dense_image_warp(a[0], a[1]))((im1, flow))

    res = convnet(im1_warped, im2, flow)
    flow_fine = res + flow

    # im1_warped_fine = tfa.image.dense_image_warp(im1, flow_fine)
    im1_warped_fine = tf.keras.layers.Lambda(lambda a: tfa.image.dense_image_warp(a[0], a[1]))((im1, flow_fine))
    
    loss_layer = tf.math.reduce_mean(tf.math.squared_difference(im1_warped_fine, im2))

    return loss_layer, flow_fine


def optical_flow(im1_4, im2_4, bach):

    im1_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_4)
    im1_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_3)
    im1_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_2)
    im1_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_1)

    im2_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_4)
    im2_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_3)
    im2_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_2)
    im2_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_1)

    # print(bach)
    flow_zero = tf.zeros((bach, im1_0.shape[1], im1_0.shape[2], 2), dtype=tf.float32)
    # flow_zero = tf.zeros_like(im1_0)
    # flow_zero = tf.zeros_like(im2_0)

    loss_0, flow_0 = loss(flow_zero, im1_0, im2_0)
    loss_1, flow_1 = loss(flow_0, im1_1, im2_1)
    loss_2, flow_2 = loss(flow_1, im1_2, im2_2)
    loss_3, flow_3 = loss(flow_2, im1_3, im2_3)
    loss_4, flow_4 = loss(flow_3, im1_4, im2_4)
  
    return flow_4
    # return im2_0
    

# MV_analysis defaults

def encoder(input, num_filters=128, kernel_size=3, M=2):
    x = tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, activation=tfc.GDN())(input)
    x = tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    x = tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    x = tfc.SignalConv2D(M, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    
    return x


def decoder(input, num_filters=128, kernel_size=3, M=2):
    x = tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(input)
    x = tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = tfc.SignalConv2D(M, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    return x




def resblock(input, IC, OC, name):

    l1 = tf.nn.relu(input, name=name + 'relu1')

    l1 = Conv2D(filters= np.minimum(IC, OC), kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(), name=name + 'l1')(l1)

    l2 = tf.nn.relu(l1, name='relu2')

    l2 = Conv2D(filters=OC, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(), name=name + 'l2')(l2)

    if IC != OC:
        input = Conv2D(filters=OC, kernel_size=1, strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(), name=name + 'map')(input)

    return input + l2


def MotionCompensation(input):

    m1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc1')(input)

    m2 = resblock(m1, 64, 64, name='mc2')

    m3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(m2)

    m4 = resblock(m3, 64, 64, name='mc4')

    m5 = AveragePooling2D(pool_size=2, strides=2, padding='same')(m4)

    m6 = resblock(m5, 64, 64, name='mc6')

    m7 = resblock(m6, 64, 64, name='mc7')

    m8 = tf.image.resize(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])

    m8 = m4 + m8

    m9 = resblock(m8, 64, 64, name='mc9')

    m10 = tf.image.resize(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

    m10 = m2 + m10

    m11 = resblock(m10, 64, 64, name='mc11')

    m12 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc12')(m11)

    m12 = tf.nn.relu(m12, name='relu12')

    m13 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc13')(m12)

    return m13

