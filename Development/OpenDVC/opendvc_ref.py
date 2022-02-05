
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
