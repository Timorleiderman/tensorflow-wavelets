import tensorflow_compression as tfc
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras import Model
import load

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

Width = 240
Height = 240
Channels = 3
batch_size = 4
frames = 10
folder = ["/workspaces/tensorflow-wavelets/Development/OpenDVC/BasketballPass"]


inp = tf.keras.Input(shape=(Width, Height, Channels,))
noisy = tfc.NoisyNormal(loc=0., scale=0.5)
entropy_quantizer_mv = tfc.ContinuousBatchedEntropyModel(noisy, 1, compression=True)


x = encoder(inp, num_filters=128, kernel_size=3, M=128)
x_hat, likelihoods = entropy_quantizer_mv(x, training=True)
y = decoder(x_hat, num_filters=128, kernel_size=3, M=2)
y_warp = tf.keras.layers.Lambda(lambda a: tfa.image.dense_image_warp(a[0], a[1]))((inp, y))


model = Model(inputs=inp, outputs=[y_warp, likelihoods], name="MyModel")
model.summary()

train_opt = tf.optimizers.Adam(learning_rate=1e-4)

data = np.zeros([frames, batch_size, Height, Width, Channels])
data - load.load_local_data(data, frames, batch_size, Height, Width, Channels, folder)

checkpoint = tf.train.Checkpoint(optimizer=train_opt, model=model)

with tf.GradientTape() as tape:
    predictions, likelihoods = model(data[0], training=True)
    train_bpp = tf.math.reduce_sum(tf.math.log(likelihoods)) / (-np.log(2) * Height * Width * batch_size)
    total_mse = tf.math.reduce_mean(tf.math.squared_difference(data[0], predictions))
    train_loss_total = 256 * total_mse + train_bpp

gradients = tape.gradient(train_loss_total, model.trainable_variables)
train_opt.apply_gradients(zip(gradients, model.trainable_variables))

print("Hey, ...")

