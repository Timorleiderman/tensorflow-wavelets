import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Remove images to get smaller dataset
x_train = x_train[:1000, :, :]
y_train = y_train[:1000]
x_test = x_test[:500, :, :]
y_test = y_test[:500]

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# CNN -> bachnorm -> relu (common structure)
# x10


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()

        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        print(x.shape)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()

        self.block1 = CNNBlock(32)
        self.block2 = CNNBlock(64)
        self.block3 = CNNBlock(128)
        self.flat = layers.Flatten()
        self.dense = layers.Dense(num_classes)

    def call(self, input_tensor, training=False, mask=None):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.flat(x)
        x = self.dense(x)
        return x

    def model(self):

        x = keras.Input(shape=(28, 28, 1))
        return keras.Model(inputs=[x], outputs=self.call(x))


model = MyModel(num_classes=10)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=2)
print(model.model().summary())
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
