import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Conv2D
import tensorflow_addons as tfa
from tensorflow.keras import layers


def convnet(im1_warp, im2, flow):

    input = tf.concat([im1_warp, im2, flow], axis=-1)

    conv1 = Conv2D(filters=32, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(input)
    conv2 = Conv2D(filters=64, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(conv1)
    conv3 = Conv2D(filters=32, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(conv2)
    conv4 = Conv2D(filters=16, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(conv3)
    conv5 = Conv2D(filters=2 , kernel_size=[7, 7], padding="same", activation=None)(conv4)

    return conv5


def loss(flow_course, im1, im2):

    flow = tf.image.resize(flow_course, [tf.shape(im1)[1], tf.shape(im2)[2]])

    im1_warped = tfa.image.dense_image_warp(im1, flow )
    res = convnet(im1_warped, im2, flow)
    flow_fine = res + flow

    im1_warped_fine = tfa.image.dense_image_warp(im1, flow_fine)
    loss_layer = tf.math.reduce_mean(tf.math.squared_difference(im1_warped_fine, im2))

    return loss_layer, flow_fine


def optical_flow(im1_4, im2_4,bach, h, w):

    im1_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_4)
    im1_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_3)
    im1_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_2)
    im1_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_1)

    im2_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_4)
    im2_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_3)
    im2_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_2)
    im2_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_1)

    flow_zero = tf.zeros((1, h//16, w//16, 2), dtype=tf.dtypes.float32)

    loss_0, flow_0 = loss(flow_zero, im1_0, im2_0)
    loss_1, flow_1 = loss(flow_0, im1_1, im2_1)
    loss_2, flow_2 = loss(flow_1, im1_2, im2_2)
    loss_3, flow_3 = loss(flow_2, im1_3, im2_3)
    loss_4, flow_4 = loss(flow_3, im1_4, im2_4)

    return flow_4, loss_0, loss_1, loss_2, loss_3, loss_4
    # return im2_0
    

class OpticalFlow(layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def build(self, input_shape):

        self.bach = input_shape[0][0]
        self.h = input_shape[0][1]
        self.w = input_shape[1][2]

        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        flow_4, loss_0, loss_1, loss_2, loss_3, loss_4 = optical_flow(inputs[0], inputs[1], self.bach, self.h, self.w)
        return flow_4

