
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, cifar10

class ResnetBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetBlock, self).__init__(name='resnet_block')

        filter1, filter2, filter3 = filters

        # Three sub-layers, each batch a convolution plus a regularization
        # First sub-layer, a convolution * 1
        self.conv1 = tf.keras.layers.Conv2D(filter1, (1,1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        # The second sub-layer, characterized kernel_size
        self.conv2 = tf.keras.layers.Conv2D(filter2, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # The third sub-layer, 1 * 1 convolution
        self.conv3 = tf.keras.layers.Conv2D(filter3, (1,1))
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):

        #Each sub-stack layer #
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        # Residual connection
        x += inputs
        outputs = tf.nn.relu(x)

        return outputs

resnetBlock = ResnetBlock(2, [6,4,9])
# Test data
print(resnetBlock(tf.ones([1,3,9,9])))
# View variable names in the network
print([x.name for x in resnetBlock.trainable_variables])