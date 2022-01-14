import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_compression as tfc
import motion
import numpy as np
import load

from tensorflow.keras.layers import AveragePooling2D, Conv2D

tf.executing_eagerly()



class ResBlock(tf.keras.layers.Layer):
    def __init__(self, IC, OC, name, **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        self.residual = tf.keras.Sequential([
            Conv2D(filters=np.minimum(IC, OC), 
                    kernel_size=3, strides=1,
                    padding='same', 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(), 
                    activation='relu',
                    name=name + 'l1'),
            Conv2D(filters=OC,
                    kernel_size=3, 
                    strides=1,
                    padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    activation='relu',
                    name=name + 'l2')
                    ])
    def call(self, inputs, training=None, mask=None):
        return self.residual(inputs)




class MotionCompensation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MotionCompensation, self).__init__(**kwargs)

        self.m1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc1')

        self.m2 = ResBlock(64, 64, name='mc2')

        self.m3 = AveragePooling2D(pool_size=2, strides=2, padding='same')

        self.m4 = ResBlock(64, 64, name='mc4')

        self.m5 = AveragePooling2D(pool_size=2, strides=2, padding='same')

        self.m6 = ResBlock(64, 64, name='mc6')

        self.m7 = ResBlock(64, 64, name='mc7')

        self.m9 = ResBlock(64, 64, name='mc9')

        self.m11 = ResBlock(64, 64, name='mc11')

        self.m12 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc12', activation='relu')

        self.m13 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.keras.initializers.GlorotUniform(), name='mc13', activation='relu')

    def call(self, inputs, training=None, mask=None):

        m1 = self.m1(inputs)
        m2 = self.m2(m1)
        m3 = self.m3(m2)
        m4 = self.m4(m3)
        m5 = self.m5(m4)
        m6 = self.m6(m5)
        m7 = self.m7(m6)

        m8 = tf.image.resize(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])
        m8 = m4 + m8
        m9 = self.m9(m8)

        m10 = tf.image.resize(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

        m10 = m2 + m10
        m11 = self.m11(m10)
        m12 = self.m12(m11)
        return self.m13(m12)



class OpticalFlowConvert(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OpticalFlowConvert, self).__init__(**kwargs)
        self.converter = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(7, 7), padding="same", activation='relu'),
            Conv2D(filters=64, kernel_size=(7, 7), padding="same", activation='relu'),
            Conv2D(filters=32, kernel_size=(7, 7), padding="same", activation='relu'),
            Conv2D(filters=16, kernel_size=(7, 7), padding="same", activation='relu'),
            Conv2D(filters=2, kernel_size=(7, 7), padding="same", activation='relu')
            ])


    def call(self, inputs, training=None, mask=None):
        # input = tf.concat([im1_warp, im2, flow], axis=-1)
        res = self.converter(inputs)

        return res

class OpticalFlowLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OpticalFlowLoss, self).__init__(**kwargs)
        self.convert = OpticalFlowConvert()

    def call(self, inputs, training=None, mask=None):
        flow_course = inputs[0]
        im1 = inputs[1]
        im2 = inputs[2]

        flow = tf.image.resize(flow_course, [tf.shape(im1)[1], tf.shape(im2)[2]])
        im1_warped = tf.keras.layers.Lambda(lambda a: tfa.image.dense_image_warp(a[0], a[1]))((im1, flow))
        convnet_input = tf.concat([im1_warped, im2, flow], axis=-1)
        res = self.convert(convnet_input)

        flow_fine = res + flow
        im1_warped_fine = tf.keras.layers.Lambda(lambda a: tfa.image.dense_image_warp(a[0], a[1]))((im1, flow_fine))
        loss_layer = tf.math.reduce_mean(tf.math.squared_difference(im1_warped_fine, im2))

        return loss_layer, flow_fine

class OpticalFlow(tf.keras.layers.Layer):
    """ 
    """
    def __init__(self, **kwargs):
        super(OpticalFlow, self).__init__(**kwargs)
        self.optic_loss = OpticalFlowLoss()

    def build(self, input_shape):
        # create filter matrix
        self.batch_size = 4

    def call(self, inputs, training=None, mask=None):
        
        im1_4 = inputs[0]
        im2_4 = inputs[1]
        im1_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_4)
        im1_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_3)
        im1_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_2)
        im1_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_1)

        im2_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_4)
        im2_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_3)
        im2_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_2)
        im2_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_1)
        
        flow_zero = tf.zeros((4, im1_0.shape[1], im1_0.shape[2], 2), dtype=tf.float32)
        
        loss_0, flow_0 = self.optic_loss([flow_zero, im1_0, im2_0])
        loss_1, flow_1 = self.optic_loss([flow_0, im1_1, im2_1])
        loss_2, flow_2 = self.optic_loss([flow_1, im1_2, im2_2])
        loss_3, flow_3 = self.optic_loss([flow_2, im1_3, im2_3])
        loss_4, flow_4 = self.optic_loss([flow_3, im1_4, im2_4])

        return flow_4


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
        
        self.optical_flow = OpticalFlow()
        self.motion_comensation = MotionCompensation()
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
        
        # flow_tensor = motion.optical_flow(Y0_com, Y1_raw, self.batch_size)
        flow_tensor = self.optical_flow([Y0_com, Y1_raw])
        # flow_tensor =  tf.keras.layers.Conv2D(filters=2, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)(Y0_com)
        flow_latent = self.mv_analysis_transform(flow_tensor)
        flow_latent_hat, MV_likelihoods_bits = entropy_model_mv(flow_latent, training=training)
        flow_hat = self.mv_synthesis_transform(flow_latent_hat)

        Y1_warp = tfa.image.dense_image_warp(Y0_com, flow_hat )

        MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)

        Y1_MC = self.motion_comensation(MC_input)

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

        return  train_loss_total, train_loss_MV, train_loss_MC, total_mse, warp_mse, MC_mse, psnr

    def train_step(self, x):
        print("Train step")
        with tf.GradientTape() as tape:
            train_loss_total, train_loss_MV, train_loss_MC, total_mse, warp_mse, MC_mse, psnr = self(x, training=True)
        
        variables = self.trainable_variables

        gradients = tape.gradient(train_loss_total, variables)
        print(gradients)
        self.train_MV_opt.apply_gradients(zip(gradients, variables))

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
    model.summary()
    model.compile()

    import load

    folder = ["/workspaces/tensorflow-wavelets/Development/OpenDVC/BasketballPass"]

    batch_size = 4
    Height = 240
    Width = 240
    Channel = 3
    lr_init = 1e-4
    frames=20
    I_QP=27

    data = np.zeros([frames, batch_size, Height, Width, Channel])
    data - load.load_local_data(data, frames, batch_size, Height, Width, Channel, folder)
    model.fit(data, epochs=15, steps_per_epoch=1, verbose=1, )    
   