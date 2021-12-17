import tensorflow as tf
import motion
import tensorflow_compression as tfc
import tensorflow_addons as tfa


class DVC(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.h = 0
        self.w = 0
        self.batch = 0
        self.noisy = tfc.NoisyNormal(loc=.5, scale=8.)
        self.entropy_quantizer_mv = tfc.ContinuousBatchedEntropyModel(noisy, 1, compression=True)


    def build(self, input_shape):
        self.h = int(input_shape[0][1])
        self.w = int(input_shape[0][2])
        self.batch = int(input_shape[0][0])
        return super().build(input_shape)

    def call(self, inputs):

        vt, _, _, _, _, _ = motion.optical_flow(inputs[0], inputs[1], 1, self.h, self.w)
        mt = motion.encoder(vt, num_filters=128, kernel_size=3, M=128)
        mt_hat, MV_likelihoods = self.entropy_quantizer_mv(mt, training=True)
        vt_hat = motion.decoder(mt_hat, num_filters=128, kernel_size=3, M=2)
        Y1_warp = tfa.image.dense_image_warp(inputs[0], vt_hat )
