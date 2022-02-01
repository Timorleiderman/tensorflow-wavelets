import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_compression as tfc
import motion
import numpy as np
import load
import resource

from tensorflow.keras.layers import AveragePooling2D, Conv2D





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

        # print("in optical loss **** im1,2 ", im1.shape, im2.shape, flow_course.shape)
        flow = tf.image.resize(flow_course, [tf.shape(im1)[1], tf.shape(im1)[2]])
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
    def __init__(self, batch_size, width, height,  **kwargs):
        super(OpticalFlow, self).__init__(**kwargs)
        self.optic_loss = OpticalFlowLoss()
        self.batch_size = batch_size
        self.width = width
        self.height = height

    def call(self, inputs, training=None, mask=None):
        
        im1_4 = inputs[0]
        im2_4 = inputs[1]
        # print("im1/2_4 avarage pooling input ", im1_4.shape, im2_4.shape)
        im1_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_4)
        im1_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_3)
        im1_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_2)
        im1_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_1)

        im2_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_4)
        im2_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_3)
        im2_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_2)
        im2_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im2_1)
        
        flow_zero = tf.zeros((self.batch_size, self.width, self.height, 2), dtype=tf.float32)

        loss_0, flow_0 = self.optic_loss([flow_zero, im1_0, im2_0])
        loss_1, flow_1 = self.optic_loss([flow_0, im1_1, im2_1])
        loss_2, flow_2 = self.optic_loss([flow_1, im1_2, im2_2])
        loss_3, flow_3 = self.optic_loss([flow_2, im1_3, im2_3])
        loss_4, flow_4 = self.optic_loss([flow_3, im1_4, im2_4])

        return flow_4


class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters=128, kernel_size=3, M=2, name="analysis"):
    super(AnalysisTransform, self).__init__(name=name)
    self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, name="layer_0",  activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, name="layer_1", activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True, name="layer_2", activation=tfc.GDN(name="gdn_2")))
    self.add(tfc.SignalConv2D(M, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros", use_bias=True,name="layer_3", activation=tfc.GDN(name="gdn_3")))
    

class SynthesisTransform(tf.keras.Sequential):
    """The synthesis transform."""
    def __init__(self, num_filters=128, kernel_size=3, M=2, name="synthesis"):
        super(SynthesisTransform, self).__init__(name=name)
        self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", name="layer_0", use_bias=True, activation=tfc.GDN(name="igdn_0", inverse=True)))
        self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", name="layer_1", use_bias=True, activation=tfc.GDN(name="igdn_1", inverse=True)))
        self.add(tfc.SignalConv2D(num_filters, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", name="layer_2", use_bias=True, activation=tfc.GDN(name="igdn_2", inverse=True)))
        self.add(tfc.SignalConv2D(M, (kernel_size, kernel_size), corr=False, strides_up=2, padding="same_zeros", name="layer_3", use_bias=True, activation=tfc.GDN(name="igdn_3", inverse=True)))

class OpenDVC(tf.keras.Model):
    """Main model class."""

    def __init__(self, width=240, height=240, batch_size=4, num_filters=128, lmbda=512):
        super(OpenDVC, self).__init__()
        self.mv_analysis_transform = AnalysisTransform(num_filters, kernel_size=3, M=128, name="mv_analysis")
        self.mv_synthesis_transform = SynthesisTransform(num_filters, kernel_size=3, name="mv_synthesis")
        self.res_analysis_transform = AnalysisTransform(num_filters, kernel_size=5, M=128, name="res_analysis")
        self.res_synthesis_transform = SynthesisTransform(num_filters, kernel_size=5, M=3, name="res_synthesis")

        self.prior_mv = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
        self.prior_res = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))

        self.optical_flow = OpticalFlow(batch_size, width, height)
        self.motion_comensation = MotionCompensation()
        self.width = width
        self.height = height
        self.batch_size = batch_size

        self.lmbda = lmbda
        # self.train_step_cnt = 0
        self.build([(None, width, height, 3),(None, width, height, 3)])

    def call(self, x, training):
        """Computes rate and distortion losses."""
        
        # Reference frame frame
        Y0_com = tf.cast(x[0], dtype=tf.float32)
        # current frame
        Y1_raw = tf.cast(x[1], dtype=tf.float32)
        # print(Y1_raw.shape)
        # print("call OpenDVC with ", Y0_com.shape, Y1_raw.shape, training)
        entropy_model_mv = tfc.ContinuousBatchedEntropyModel(self.prior_mv, coding_rank=3, compression=False)
        entropy_model_res = tfc.ContinuousBatchedEntropyModel(self.prior_res, coding_rank=3, compression=False)

        flow_tensor = self.optical_flow([Y0_com, Y1_raw])
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


        # bpp
        # train_bpp_MV = tf.reduce_sum(tf.math.log(MV_likelihoods_bits)) / (-np.log(2) * self.height * self.width * self.batch_size)
        # train_bpp_Res = tf.reduce_sum(tf.math.log(Res_likelihoods_bits)) / (-np.log(2) * self.height * self.width * self.batch_size)

        num_pixels = tf.cast(tf.reduce_prod(tf.shape(Y0_com)[:-1]), MV_likelihoods_bits.dtype)
        bpp = ( tf.reduce_sum(MV_likelihoods_bits) + tf.reduce_sum(Res_likelihoods_bits) ) /  num_pixels

        # bpp = ( tf.reduce_sum(MV_likelihoods_bits) ) /  num_pixels
        # mse
        mse = tf.reduce_mean(tf.math.squared_difference(Y1_com, Y1_raw))
        # mse = tf.reduce_mean(tf.math.squared_difference(Y1_warp, Y1_raw))
        # mse = tf.reduce_mean(tf.math.squared_difference(Y1_raw, Y1_MC))
        # ME_mse = tf.reduce_mean(tf.math.squared_difference(Y1_com, Y1_raw))
        # warp_mse = tf.reduce_mean(tf.math.squared_difference(Y1_warp, Y1_raw))
        # MC_mse = tf.reduce_mean(tf.math.squared_difference(Y1_raw, Y1_MC))

        # psnr = 10.0*tf.math.log(1.0/ME_mse)/tf.math.log(10.0)
        
        # loss
        # train_loss_ME = self.l * ME_mse + (train_bpp_MV + train_bpp_Res)
        # train_loss_MV = self.l * warp_mse + train_bpp_MV
        # train_loss_MC = self.l * MC_mse + train_bpp_MV

        loss =  bpp + self.lmbda * mse
        # return  train_loss_ME, train_loss_MV, train_loss_MC, ME_mse, warp_mse, MC_mse, train_bpp_MV, train_bpp_Res, psnr
        return  loss, bpp, mse

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

    def test_step(self, x):
        loss, bpp, mse = self(x, training=False)
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

    # def train_step(self, x):
    #     # print("Train iter", self.iter)
        
    #     with tf.GradientTape() as tape:
    #         train_loss_ME, train_loss_MV, train_loss_MC, ME_mse, warp_mse, MC_mse, train_bpp_MV, train_bpp_Res, psnr = self(x, training=True)
            
    #     variables = self.trainable_variables
        
    #     if self.iter < 1000:
    #         loss = train_loss_MV
    #         print("MV loss")
    #     elif self.iter < 2000:
    #         loss = train_loss_MC
    #         print("MC loss")
    #     else:
    #         loss = train_loss_ME
    #         print("total loss")

    #     gradients = tape.gradient(loss, variables)
    #     self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, variables) if grad is not None)

    #     self.train_loss_ME.update_state(train_loss_ME)
    #     self.ME_mse.update_state(ME_mse)

    #     self.train_loss_MV.update_state(train_loss_MV)
    #     self.warp_mse.update_state(warp_mse)

    #     self.train_loss_MC.update_state(train_loss_MC)
    #     self.MC_mse.update_state(MC_mse)

    #     self.train_bpp_MV.update_state(train_bpp_MV)
    #     self.train_bpp_Res.update_state(train_bpp_Res)

    #     self.psnr.update_state(psnr)

    #     self.iter += 1
    #     # self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, variables) if grad is not None)
    #     # with tf.GradientTape() as tape_mv:
    #     #     train_loss_ME, train_loss_MV, train_loss_MC, ME_mse, warp_mse, MC_mse, psnr = self(x, training=True)
        
    #     # variables = self.trainable_variables
    #     # gradients = tape_mv.gradient(train_loss_MV, variables)
    #     # self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, variables) if grad is not None)

    #     # with tf.GradientTape() as tape_mc:
    #     #     train_loss_ME, train_loss_MV, train_loss_MC, ME_mse, warp_mse, MC_mse, psnr = self(x, training=True)

    #     # variables = self.trainable_variables
    #     # gradients = tape_mc.gradient(train_loss_MC, variables)
    #     # self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, variables) if grad is not None)

    
    #     return {m.name: m.result() for m in [self.train_loss_ME, self.train_loss_MV, self.train_loss_MC, self.ME_mse, self.warp_mse, self.MC_mse, self.train_bpp_MV, self.train_bpp_Res, self.psnr]}
    # def test_step(self, x):

    #     train_loss_ME, train_loss_MV, train_loss_MC, ME_mse, warp_mse, MC_mse, train_bpp_MV, train_bpp_Res, psnr = self(x, training=False)

    #     self.train_loss_ME.update_state(train_loss_ME)
    #     self.ME_mse.update_state(ME_mse)

    #     self.train_loss_MV.update_state(train_loss_MV)
    #     self.warp_mse.update_state(warp_mse)

    #     self.train_loss_MC.update_state(train_loss_MC)
    #     self.MC_mse.update_state(MC_mse)

    #     self.train_bpp_MV.update_state(train_bpp_MV)
    #     self.train_bpp_Res.update_state(train_bpp_Res)
        
    #     self.psnr.update_state(psnr)
        
    #     return {m.name: m.result() for m in  [self.train_loss_ME, self.train_loss_MV, self.train_loss_MC, self.ME_mse, self.warp_mse, self.MC_mse, self.train_bpp_MV, self.train_bpp_Res, self.psnr]}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(240, 240, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(240, 240, 3), dtype=tf.uint8),
    ])
    def compress(self, Y0_com, Y1_raw):
        """Compresses an image."""
        # Add batch dimension and cast to float.
        print("in the compress")
        Y0_com = tf.expand_dims(Y0_com, 0)
        Y1_raw = tf.expand_dims(Y1_raw, 0)
        Y0_com = tf.cast(Y0_com, dtype=tf.float32)
        Y1_raw = tf.cast(Y1_raw, dtype=tf.float32)

        flow_tensor = self.optical_flow([Y0_com, Y1_raw])
        # print("flow_tensor ", flow_tensor.shape)
        flow_latent = self.mv_analysis_transform(flow_tensor)
        flow_latent_hat, _ = self.entropy_model_mv(flow_latent, training=False)
        flow_hat = self.mv_synthesis_transform(flow_latent_hat)

        Y1_warp = tfa.image.dense_image_warp(Y0_com, flow_hat )
        MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
        Y1_MC = self.motion_comensation(MC_input)
        Res = Y1_raw - Y1_MC
        res_latent = self.res_analysis_transform(Res)
        res_latent_hat, _ = self.entropy_model_res(res_latent, training=False)
        
        # Res_hat = self.res_synthesis_transform(res_latent_hat)
        # Y1_com = Res_hat + Y1_MC

        # Preserve spatial shapes of both image and latents.
        x_shape = tf.shape(Y0_com)[1:-1]
        y_shape = tf.shape(flow_latent)[1:-1]
        z_shape = tf.shape(res_latent)[1:-1]

        mv_str_bits = self.entropy_model_mv.compress(flow_latent)
        res_str_bits = self.entropy_model_res.compress(res_latent)
        return mv_str_bits, res_str_bits, x_shape, y_shape, z_shape

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(240, 240, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(1,), dtype=tf.string),
        tf.TensorSpec(shape=(1,), dtype=tf.string),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
    ])
    def decompress(self, ref_frame, mv_str_bits, res_str_bits, x_shape, y_shape, z_shape):
        """Decompresses an image."""
        print("in decompress")
        ref_frame = tf.expand_dims(ref_frame, 0)
        Y0_com = tf.cast(ref_frame, dtype=tf.float32)

        flow_latent_hat = self.entropy_model_mv.decompress(mv_str_bits, y_shape)
        flow_hat = self.mv_synthesis_transform(flow_latent_hat)
        Y1_warp = tfa.image.dense_image_warp(Y0_com, flow_hat )
        MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
        Y1_MC = self.motion_comensation(MC_input)
        res_latent_hat = self.entropy_model_res.decompress(res_str_bits, z_shape)
        Res_hat = self.res_synthesis_transform(res_latent_hat)
        Y1_dcom = Res_hat + Y1_MC

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = Y1_dcom[0, :x_shape[0], :x_shape[1], :]
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)

    # def compile(self, freeze, **kwargs):
    #     super().compile(loss=None, metrics=None, loss_weights=None, weighted_metrics=None, **kwargs,)
        
    #     self.train_loss_ME = tf.keras.metrics.Mean(name="train_loss_ME")
    #     self.train_loss_MV = tf.keras.metrics.Mean(name="train_loss_MV")
    #     self.train_loss_MC = tf.keras.metrics.Mean(name="train_loss_MC")

    #     self.psnr = tf.keras.metrics.Mean(name="psnr")
    #     self.ME_mse = tf.keras.metrics.Mean(name="ME_mse")
    #     self.warp_mse = tf.keras.metrics.Mean(name="warp_mse")
    #     self.MC_mse = tf.keras.metrics.Mean(name="MC_mse")

    #     self.train_bpp_MV = tf.keras.metrics.Mean(name="bpp_MV")
    #     self.train_bpp_Res = tf.keras.metrics.Mean(name="bpp_Res")

    #     self.iter = 0
    #     for layer_idx in freeze:
    #         self.layers[layer_idx].trainable = False
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
        self.entropy_model_mv = tfc.ContinuousBatchedEntropyModel(
            self.prior_mv, coding_rank=3, compression=True)

        self.entropy_model_res = tfc.ContinuousBatchedEntropyModel(
            self.prior_res, coding_rank=3, compression=True)

        return retval

    def predict_step(self, x):
        raise NotImplementedError("Prediction API is not supported.")

def read_png_resize(filename, width, height):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    img_res = tf.image.resize(image, [width,height])
    return tf.cast(img_res, dtype=tf.uint8)

def read_png_crop(filename, width, height):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    img_crop = tf.image.crop_to_bounding_box(image, 0, 0, width, height)
    return tf.cast(img_crop, dtype=tf.uint8)

def write_png(filename, image):
    """Saves an image to a PNG file."""
    string = tf.image.encode_png(image)
    tf.io.write_file(filename, string)


def compress(model, input_i, input_p, output_bin, width=240, height=240):
    # model = tf.keras.models.load_model(args.model_path)
    print("compress")
    Y0_com = read_png_crop(input_i, width, height)
    Y1_raw = read_png_crop(input_p, width, height)

    tensors = model.compress(Y0_com, Y1_raw)
    
    packed = tfc.PackedTensors()
    packed.pack(tensors)
    with open(output_bin, "wb") as f:
        f.write(packed.string)


def decompress(model, input_ref, input_bin, output_decom, width=240, height=240):
    """Decompresses an image."""
    # Load the model and determine the dtypes of tensors required to decompress.
    # model = tf.keras.models.load_model(args.model_path)
    dtypes = [t.dtype for t in model.decompress.input_signature[1:]]

    Y1_Ref = read_png_crop(input_ref, width, height)

    # Read the shape information and compressed string from the binary file,
    # and decompress the image using the model.
    with open(input_bin, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = packed.unpack(dtypes)
    tensors.insert(0, Y1_Ref)
    x_hat = model.decompress(*tensors)

    # Write reconstructed image out as a PNG file.
    write_png(output_decom, x_hat)

class Arguments(object):
    def __init__(self) -> None:
        super().__init__()

        self.model_checkpoints = "checkpoint/"
        self.model_checkpoints_me = "checkpoint_me/"
        self.model_checkpoints_mv = "checkpoint_mv/"
        self.model_checkpoints_mc = "checkpoint_mc/"
        self.model_save = "model_save/1/"
        self.backup_restore = "backup/"


class MemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print("[MemoryCallback]: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":

    import load
    tf.executing_eagerly()
    folder = ["/workspaces/tensorflow-wavelets/Development/OpenDVC/BasketballPass"]

    batch_size = 1
    Height = 240
    Width = 240
    Channel = 3
    lr_init = 1e-4
    frames=100
    I_QP=27
    epochs=25
    checkpoint_filepath = "checkpoint/"
    backup_restore = "backup/"
    model_save = "model_save/1/"

    model = OpenDVC(width=Width, height=Height, batch_size=batch_size, num_filters=128)
    # model.summary()
    model.compile()


    data = np.zeros([frames, batch_size, Height, Width, Channel])
    data - load.load_local_data(data, frames, batch_size, Height, Width, Channel, folder)
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(2)
    # for elem in dataset:
    #     print(elem.shape)
    
    model.fit(dataset,
              epochs=epochs, 
              verbose=1, 
              batch_size=2,
              callbacks=
              [
                MemoryCallback(),
                tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath, save_weights_only=True, save_freq='epoch', monitor='train_loss_total', mode='max', save_best_only=True), 
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.TensorBoard(log_dir=backup_restore, histogram_freq=0, update_freq="epoch"),
                tf.keras.callbacks.experimental.BackupAndRestore(backup_restore),
                ],
                )   
    print("Done Training ...")
    tf.saved_model.save(model, model_save)
    print("saved", model_save)
    # compress(Arguments())