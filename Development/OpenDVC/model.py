import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_compression as tfc
import motion
import numpy as np
import load

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

    def __init__(self, width=240, height=240, batch_size=4, num_filters=128):
        super().__init__()
        self.mv_analysis_transform = AnalysisTransform(num_filters, kernel_size=3, M=128)
        self.mv_synthesis_transform = SynthesisTransform(num_filters, kernel_size=3)
        self.res_analysis_transform = AnalysisTransform(num_filters, kernel_size=5, M=128)
        self.res_synthesis_transform = SynthesisTransform(num_filters, kernel_size=5, M=3)
        self.prior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
        

        self.width = width
        self.height = height
        self.batch_size = batch_size

        self.l = 256
        self.build((2, batch_size, width, height, 3))


    def call(self, x, training):
        """Computes rate and distortion losses."""

        Y0_com = x[0]
        Y1_raw = x[1]

        entropy_model_mv = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=3, compression=False)
        entropy_model_res = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=3, compression=False)

        # optical flow
        
        flow_tensor = motion.optical_flow(Y0_com, Y1_raw, self.batch_size)
        flow_latent = self.mv_analysis_transform(flow_tensor)
        flow_latent_hat, MV_likelihoods_bits = entropy_model_mv(flow_latent, training=training)
        flow_hat = self.mv_synthesis_transform(flow_latent_hat)

        Y1_warp = tfa.image.dense_image_warp(Y0_com, flow_hat )
        MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
        Y1_MC = motion.MotionCompensation(MC_input)

        Res = Y1_raw - Y1_MC

        res_latent = self.res_analysis_transform(Res)
        res_latent_hat, Res_likelihoods_bits = entropy_model_res(res_latent, training=training)
        Res_hat = self.res_synthesis_transform(res_latent_hat)
        Y1_com = Res_hat + Y1_MC

        train_bpp_MV = tf.reduce_sum(tf.math.log(MV_likelihoods_bits)) / (-np.log(2) * self.height * self.width * self.batch_size)
        train_bpp_Res = tf.reduce_sum(tf.math.log(Res_likelihoods_bits)) / (-np.log(2) * self.height * self.width * self.batch_size)

        total_mse = tf.reduce_mean(tf.math.squared_difference(Y1_com, Y1_raw))
        warp_mse = tf.reduce_mean(tf.math.squared_difference(Y1_warp, Y1_raw))
        MC_mse = tf.reduce_mean(tf.math.squared_difference(Y1_raw, Y1_MC))

        psnr = 10.0*tf.math.log(1.0/total_mse)/tf.math.log(10.0)
        
        train_loss_total = self.l * total_mse + (train_bpp_MV + train_bpp_Res)
        train_loss_MV = self.l * warp_mse + train_bpp_MV
        train_loss_MC = self.l * MC_mse + train_bpp_MV

        return train_loss_total, train_loss_MV, train_loss_MC, total_mse, warp_mse, MC_mse, psnr

    def train_step(self, x):
        with tf.GradientTape() as tape:
            train_loss_total, train_loss_MV, train_loss_MC, total_mse, warp_mse, MC_mse, psnr = self(x, training=True)
        variables = self.trainable_variables
        gradients = tape.gradient(train_loss_total, variables)

        # self.train_MV_opt.apply_gradients(zip(gradients, variables))

        # self.train_loss_total.update_state(train_loss_total)
        # self.train_loss_MV.update_state(train_loss_MV)
        # self.train_loss_MC.update_state(train_loss_MC)
        # self.psnr.update_state(psnr)
        # self.total_mse.update_state(total_mse)
        # self.warp_mse.update_state(warp_mse)
        # self.MC_mse.update_state(MC_mse)

    
        return {m.name: m.result() for m in [self.train_loss_total, self.train_loss_MV, self.train_loss_MC, self.psnr, self.total_mse, self.warp_mse, self.MC_mse]}


    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
    ])
    def compress(self, flow_tensor):
        """Compresses an image."""
        # Add batch dimension and cast to float.
        x = tf.expand_dims(flow_tensor, 0)
        x = tf.cast(x, dtype=tf.float32)
        flow_latent = self.mv_analysis_transform(x)
        # Preserve spatial shapes of both image and latents.
        x_shape = tf.shape(x)[1:-1]
        y_shape = tf.shape(flow_latent)[1:-1]
        return self.entropy_model.compress(flow_latent), x_shape, y_shape

    def compile(self, **kwargs):
        super().compile(loss=None, metrics=None, loss_weights=None, weighted_metrics=None, **kwargs,)
        
        self.train_loss_total = tf.keras.metrics.Mean(name="train_loss_total")
        self.train_loss_MV = tf.keras.metrics.Mean(name="train_loss_MV")
        self.train_loss_MC = tf.keras.metrics.Mean(name="train_loss_MC")
        self.psnr = tf.keras.metrics.Mean(name="psnr")
        self.total_mse = tf.keras.metrics.Mean(name="total_mse")
        self.warp_mse = tf.keras.metrics.Mean(name="warp_mse")
        self.MC_mse = tf.keras.metrics.Mean(name="MC_mse")

        learning_rate = 1e-4
        self.train_MV_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_MC = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_total = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        aux_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate*10.0)
        aux_optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate*10.0)

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)
        # After training, fix range coding tables.
        self.entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.prior, coding_rank=3, compression=True)
        return retval

if __name__ == "__main__":
    print("gg")
    model = OpenDVC()
    # model.summary()
    model.compile()

    import load

    folder = ["/workspaces/tensorflow-wavelets/Development/OpenDVC/BasketballPass"]

    batch_size = 4
    Height = 240
    Width = 240
    Channel = 3
    lr_init = 1e-4
    frames=2
    I_QP=27

    data = np.zeros([frames, batch_size, Height, Width, Channel])
    data - load.load_local_data(data, frames, batch_size, Height, Width, Channel, folder)
    model.fit(data, epochs=4, steps_per_epoch=1, verbose=1, )    
   