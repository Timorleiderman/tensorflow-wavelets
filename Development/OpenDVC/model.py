import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_compression as tfc
import motion

class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters=128, kernel_size=3, M=2):
    super().__init__(name="analysis")
    self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, name="layer_0",  activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, name="layer_1", activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, name="layer_2", activation=tfc.GDN(name="gdn_2")))
    self.add(tfc.SignalConv2D(M, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True,name="layer_3", activation=tfc.GDN(name="gdn_3")))
    

class SynthesisTransform(tf.keras.Sequential):
    """The synthesis transform."""
    def __init__(self, num_filters=128, kernel_size=3, M=2):
        super().__init__(name="synthesis")
        self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", name="layer_0", use_bias=True, activation=tfc.GDN(name="igdn_0", inverse=True)))
        self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", name="layer_1", use_bias=True, activation=tfc.GDN(name="igdn_1", inverse=True)))
        self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", name="layer_2", use_bias=True, activation=tfc.GDN(name="igdn_2", inverse=True)))
        self.add(tfc.SignalConv2D(M, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", name="layer_3", use_bias=True, activation=tfc.GDN(name="igdn_3", inverse=True)))


class OpenDVC(tf.keras.Model):
    """Main model class."""

    def __init__(self, num_filters=128):
        super().__init__()
        self.analysis_transform = AnalysisTransform(num_filters)
        self.synthesis_transform = SynthesisTransform(num_filters)
        self.prior = tfc.NoisyDeepFactorized(batch_shape=(2,))
        self.build((None, 240, 240, 3))

    def call(self, x, training):
        """Computes rate and distortion losses."""
        entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.prior, coding_rank=3, compression=False)
        # optical flow
        flow = motion.optical_flow(x, x, 1, 240, 240)
        y = self.analysis_transform(flow)
        y_hat, bits = entropy_model(y, training=training)
        x_hat = self.synthesis_transform(y_hat)
        x_warp = tfa.image.dense_image_warp(x, x_hat )

        # Total number of bits divided by total number of pixels.
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
        bpp = tf.reduce_sum(bits) / num_pixels
        # Mean squared error across pixels.
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_warp))
        # The rate-distortion Lagrangian.
        loss = bpp + 0.01 * mse
        return loss, bpp, mse

    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss, bpp, mse = self(x, training=True)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}


    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
    ])
    def compress(self, x):
        """Compresses an image."""
        # Add batch dimension and cast to float.
        x = tf.expand_dims(x, 0)
        x = tf.cast(x, dtype=tf.float32)
        y = self.analysis_transform(x)
        # Preserve spatial shapes of both image and latents.
        x_shape = tf.shape(x)[1:-1]
        y_shape = tf.shape(y)[1:-1]
        return self.entropy_model.compress(y), x_shape, y_shape

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.bpp = tf.keras.metrics.Mean(name="bpp")
        self.mse = tf.keras.metrics.Mean(name="mse")

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)
        # After training, fix range coding tables.
        self.entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.prior, coding_rank=3, compression=True)
        return retval

if __name__ == "__main__":
    print("gg")
    model = OpenDVC()
    model.summary()
    