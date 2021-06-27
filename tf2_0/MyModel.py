import os
import cv2
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, cifar10

import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")


class MyReLU(layers.Layer):
    def __init__(self):
        super(MyReLU, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)


class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def call(self, inputs):
        return tf.matmul(inputs, self.W) + self.b

    def build(self, input_shape):

        self.W = self.add_weight(
            name='W',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    #
    # output = Conv2D(32, (3, 3), padding="same")(inputs)
    # output = BatchNormalization()(output)
    # output = Activation("relu")(output)
    #



class MyDWTModel(keras.Model):
    def __init__(self):
        super(MyDWTModel, self).__init__()

        self.conv1 = layers.Conv2D(32, 1)
        self.pooling2d = layers.MaxPooling2D(pool_size=2)
        self.bn1 = layers.BatchNormalization()
        self.relu = MyReLU()

        self.flat = layers.Flatten()
        self.dense = layers.Dense(32)


    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.pooling2d(x)
        x = self.relu(x)
        x = self.flat(x)
        x = self.dense(x)

        return x

# custom filter
def my_filter(shape, dtype=None):

    f = tf.constant(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype="float32"
    )
    assert f.shape == shape
    return f

def kernelInitializer(shape, dtype=None):
    sobel_x = tf.constant(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype="float32"
    )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2], shape[-1]))
    return sobel_x


class MyCNNSobelModel(layers.Layer):
    def __init__(self):
        super(MyCNNSobelModel, self).__init__()

        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            #input_shape=(32, 32, 1),
            kernel_initializer=kernelInitializer,
            padding='same',
            trainable=False
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        return x


class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(num_classes)
        self.relu = MyReLU()

        # self.dense1 = layers.Dense(64)
        # self.dense2 = layers.Dense(num_classes)

    def call(self, input_tensor):
        # x = tf.nn.relu(self.dense1(input_tensor))
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)


# model = MyModel()
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )
#
# model.fit(x_train,y_train,batch_size=32, verbose=2)
# model.evaluate(x_test,y_test,batch_size=32,verbose=2)




frog = tf.expand_dims(
    x_train[0, :, :, 1], 0, name=None
)
frog = tf.expand_dims(
    frog, -1, name=None
)
print(frog.shape)

model = keras.Sequential()
model.add(keras.Input(shape=(32, 32, 1)))
model.add(MyCNNSobelModel())
model.summary()
#
a = model.predict(frog, steps=1)



import scipy.signal as sig
filter = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sig_sobel = sig.convolve2d(x_train[0, :, :, 1], filter, mode='same')
cv_sobel = cv2.filter2D(x_train[0, :, :, 1], -1, cv2.flip(filter, -1), borderType=cv2.BORDER_CONSTANT)

f, axarr = plt.subplots(1, 4)
f.set_size_inches(16, 6)


with tf.compat.v1.Session() as sess:
    ans1 = frog[0, ..., 0].eval()
    ans2 = a[0, ..., 0]

    axarr[0].imshow(ans1, interpolation='none', cmap='gray', vmin=0, vmax=255)
    axarr[1].imshow(ans2, interpolation='none', cmap='gray', vmin=0, vmax=255)
    axarr[2].imshow(cv_sobel, interpolation='none', cmap='gray', vmin=0, vmax=255)
    axarr[3].imshow(sig_sobel, interpolation='none', cmap='gray', vmin=0, vmax=255)

# # show the figure
plt.show()

#print([x.name for x in model.trainable_variables])
#
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )
# model.fit(x_train, y_train, batch_size=32, verbose=2)

# # model.evaluate(x_test, y_test, batch_size=32, verbose=2)