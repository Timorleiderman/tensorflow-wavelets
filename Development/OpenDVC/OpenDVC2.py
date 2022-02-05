from OpenDVC import *

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_compression as tfc
import motion
import numpy as np
import load


from tensorflow.keras.layers import AveragePooling2D, Conv2D

class OpenDVC_SSIM(tf.keras.Model):
    """Main model class."""

    def __init__(self, width=240, height=240, batch_size=4, num_filters=128, lmbda=512):
        super(OpenDVC_SSIM, self).__init__()
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


        #ssim
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(Y0_com)[:-1]), MV_likelihoods_bits.dtype)
        bpp = ( tf.reduce_sum(MV_likelihoods_bits) + tf.reduce_sum(Res_likelihoods_bits) ) /  num_pixels
        ssim = tf.reduce_mean(tf.image.ssim_multiscale(Y1_com, Y1_raw, 1))
        loss =  bpp + self.lmbda * (1-ssim)

        return  loss, bpp, ssim

    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss, bpp, ssim = self(x, training=True)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.ssim.update_state(ssim)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.ssim]}

    def test_step(self, x):
        loss, bpp, ssim = self(x, training=False)
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.ssim.update_state(ssim)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.ssim]}

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
        Y0_com = tf.cast(Y0_com / 255, dtype=tf.float32)
        Y1_raw = tf.cast(Y1_raw / 255, dtype=tf.float32)

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
        Y0_com = tf.cast(ref_frame / 255, dtype=tf.float32)

        flow_latent_hat = self.entropy_model_mv.decompress(mv_str_bits, y_shape)
        flow_hat = self.mv_synthesis_transform(flow_latent_hat)
        Y1_warp = tfa.image.dense_image_warp(Y0_com, flow_hat )
        MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
        Y1_MC = self.motion_comensation(MC_input)
        res_latent_hat = self.entropy_model_res.decompress(res_str_bits, z_shape)
        Res_hat = self.res_synthesis_transform(res_latent_hat)
        Y1_dcom = Res_hat + Y1_MC

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = Y1_dcom[0, :x_shape[0], :x_shape[1], :] *255
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)

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
        self.ssim = tf.keras.metrics.Mean(name="ssim")

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








class OpenDVC_MV(tf.keras.Model):
    """Main model class."""

    def __init__(self, width=240, height=240, batch_size=4, num_filters=128, lmbda=512):
        super(OpenDVC_MV, self).__init__()
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

        # bpp = ( tf.reduce_sum(MV_likelihoods_bits) + tf.reduce_sum(Res_likelihoods_bits) ) /  num_pixels

        #warp mse and bpp mse
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(Y0_com)[:-1]), MV_likelihoods_bits.dtype)
        bpp = ( tf.reduce_sum(MV_likelihoods_bits) ) /  num_pixels
        mse = tf.reduce_mean(tf.math.squared_difference(Y1_warp, Y1_raw))
        loss =  bpp + self.lmbda * mse

        # num_pixels = tf.cast(tf.reduce_prod(tf.shape(Y0_com)[:-1]), MV_likelihoods_bits.dtype)
        # bpp = tf.reduce_sum(MV_likelihoods_bits) /  num_pixels
        # mse = tf.reduce_mean(Y1_com, Y1_raw)

        # loss =  bpp + self.lmbda * mse

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
        Y0_com = tf.cast(Y0_com / 255, dtype=tf.float32)
        Y1_raw = tf.cast(Y1_raw / 255, dtype=tf.float32)

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
        Y0_com = tf.cast(ref_frame / 255, dtype=tf.float32)

        flow_latent_hat = self.entropy_model_mv.decompress(mv_str_bits, y_shape)
        flow_hat = self.mv_synthesis_transform(flow_latent_hat)
        Y1_warp = tfa.image.dense_image_warp(Y0_com, flow_hat )
        MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
        Y1_MC = self.motion_comensation(MC_input)
        res_latent_hat = self.entropy_model_res.decompress(res_str_bits, z_shape)
        Res_hat = self.res_synthesis_transform(res_latent_hat)
        Y1_dcom = Res_hat + Y1_MC

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = Y1_dcom[0, :x_shape[0], :x_shape[1], :] * 255
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)

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






class OpenDVC_NORM(tf.keras.Model):
    """Main model class."""

    def __init__(self, width=240, height=240, batch_size=4, num_filters=128, lmbda=512):
        super(OpenDVC_NORM, self).__init__()
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

        num_pixels = tf.cast(tf.reduce_prod(tf.shape(Y0_com)[:-1]), MV_likelihoods_bits.dtype)
        bpp = ( tf.reduce_sum(MV_likelihoods_bits) + tf.reduce_sum(Res_likelihoods_bits) ) /  num_pixels
        mse = tf.reduce_mean(tf.math.squared_difference(Y1_com, Y1_raw))
        loss =  bpp + self.lmbda * mse
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
        Y0_com = tf.cast(Y0_com / 255, dtype=tf.float32)
        Y1_raw = tf.cast(Y1_raw / 255, dtype=tf.float32)

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
        Y0_com = tf.cast(ref_frame / 255, dtype=tf.float32)

        flow_latent_hat = self.entropy_model_mv.decompress(mv_str_bits, y_shape)
        flow_hat = self.mv_synthesis_transform(flow_latent_hat)
        Y1_warp = tfa.image.dense_image_warp(Y0_com, flow_hat )
        MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
        Y1_MC = self.motion_comensation(MC_input)
        res_latent_hat = self.entropy_model_res.decompress(res_str_bits, z_shape)
        Res_hat = self.res_synthesis_transform(res_latent_hat)
        Y1_dcom = Res_hat + Y1_MC

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = Y1_dcom[0, :x_shape[0], :x_shape[1], :] * 255
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)

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



