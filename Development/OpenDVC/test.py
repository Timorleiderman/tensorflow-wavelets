import numpy as np
from tensorflow.python.keras.engine import training

import load
import tensorflow as tf
import motion
import numpy as np
import tensorflow_compression as tfc
import tensorflow_addons as tfa

tf.config.run_functions_eagerly(True)

batch_size = 4
Height = 256
Width = 256
Channel = 3
lr_init = 1e-4
frames=2
I_QP = 27
l = 256


folder = np.load("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy")
#a = load.load_data()
data = np.zeros([frames, batch_size, Height, Width, Channel])
data = load.load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP)
print("Data Load done! ...")

noisy = tfc.NoisyNormal(loc=.5, scale=8.)
entropy_quantizer_mv = tfc.ContinuousBatchedEntropyModel(noisy, 1, compression=True)


noisy = tfc.NoisyNormal(loc=.5, scale=8.)
entropy_quantizer_mv = tfc.ContinuousBatchedEntropyModel(noisy, 1, compression=True)
entropy_quantizer_res = tfc.ContinuousBatchedEntropyModel(noisy, 1, compression=True)

inputs_y0_com = tf.keras.Input(shape=(Width, Height, Channel,))
inputs_y1_raw = tf.keras.Input(shape=(Width, Height, Channel,))
flow_4 = motion.optical_flow(inputs_y0_com, inputs_y1_raw, batch_size, Height, Width)
mt = motion.encoder(flow_4, num_filters=128, kernel_size=3, M=128)
mt_hat, MV_likelihoods = entropy_quantizer_mv(mt, training=True)
vt_hat = motion.decoder(mt_hat, num_filters=128, kernel_size=3, M=2)
Y1_warp = tf.keras.layers.Lambda(lambda a: tfa.image.dense_image_warp(a[0], a[1]))((inputs_y0_com, vt_hat))

MC_input = tf.concat([vt_hat, inputs_y0_com, Y1_warp], axis=-1)
# shape (1, 240 416, 8)
Y1_MC = motion.MotionCompensation(MC_input)
Res = inputs_y1_raw - Y1_MC
res_latent = motion.encoder(Res, num_filters=128, kernel_size=5, M=128)
res_latent_hat, Res_likelihoods = entropy_quantizer_res(res_latent)
Res_hat = motion.decoder(res_latent_hat, num_filters=128, kernel_size=5, M=3)

# Reconstructed frame
Y1_com = Res_hat + Y1_MC
model = tf.keras.Model([inputs_y0_com, inputs_y1_raw], Y1_com)

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())


model.fit([data[0], data[1]], data[0],
                epochs=5,
                batch_size=4,)

# for epoch in range(5):
#     with tf.GradientTape() as tape:
#         logits = model([data[0], data[1]], training=True)
#         # loss_value = loss_fn(data[1], logits)

#     # grads = tape.gradient(loss_value, model.trainable_weights)


