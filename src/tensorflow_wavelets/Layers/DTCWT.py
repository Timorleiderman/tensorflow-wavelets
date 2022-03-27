from tensorflow.keras import layers
from tensorflow_wavelets.utils import filters
from tensorflow_wavelets.utils.helpers import *
from tensorflow_wavelets.utils.cast import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DTCWT(layers.Layer):
    """
    Durel Tree Complex Wavelet Transform
    Input: level - tree-level (int)
    """
    def __init__(self, level=1, concat=True, **kwargs):
        super(DTCWT, self).__init__(**kwargs)

        if level <= 1:
            level = 1

        self.level = int(level)
        self.conv_type = "SAME"
        self.border_padd = "SYMMETRIC"

        # Faf - First analysis filter - for the first level
        # Fsf - First synthesis filter
        faf, fsf = filters.fs_farras()
        af, sf = filters.duelfilt()

        # convert to tensor
        self.Faf = duel_filter_tf(faf)
        self.Fsf = duel_filter_tf(fsf)
        self.af = duel_filter_tf(af)
        self.sf = duel_filter_tf(sf)

        self.concat = concat
    def build(self, input_shape):
        # repeat last channel if input channel bigger then 1
        if input_shape[-1] > 1:
            self.Faf = tf.repeat(self.Faf, input_shape[-1], axis=-1)
            self.Fsf = tf.repeat(self.Fsf, input_shape[-1], axis=-1)
            self.af = tf.repeat(self.af, input_shape[-1], axis=-1)
            self.sf = tf.repeat(self.sf, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        # normalize
        x_norm = tf.math.divide(inputs, 2)

        # 2 trees J+1 lists
        w = [[[[], []] for _ in range(2)] for __ in range(self.level+1)]

        # 2 trees - 2 filters ( first stage is using differnet filter
        for m in range(2):
            for n in range(2):
                [lo, w[0][m][n]] = analysis_filter_bank2d(x_norm, self.Faf[m][0], self.Faf[m][1],
                                                          self.Faf[n][0], self.Faf[n][1])
                for j in range(1, self.level):
                    [lo, w[j][m][n]] = analysis_filter_bank2d(lo, self.af[m][0], self.af[m][1],
                                                              self.af[n][0], self.af[n][1])
                w[self.level][m][n] = lo

        # add and subtract for the complex
        for j in range(self.level):
            for m in range(3):

                w[j][0][0][m], w[j][1][1][m] = add_sub(w[j][0][0][m], w[j][1][1][m])
                w[j][0][1][m], w[j][1][0][m] = add_sub(w[j][0][1][m], w[j][1][0][m])

        # concat into one big image
        # different resolution as the tree is deeper
        # TODO: How to split different resolutions into different channels
        if not self.concat:
            return w
        j = 1
        w_c = w

        for j in [x for x in range(1, self.level)][::-1]:

            w_c[j][0][0] = tf.concat([tf.concat([w_c[j+1][0][0], w_c[j][0][0][0]], axis=2),
                                      tf.concat([w_c[j][0][0][1], w_c[j][0][0][2]], axis=2)], axis=1)
            w_c[j][0][1] = tf.concat([tf.concat([w_c[j+1][0][1], w_c[j][0][1][0]], axis=2),
                                      tf.concat([w_c[j][0][1][1], w_c[j][0][1][2]], axis=2)], axis=1)
            w_c[j][1][0] = tf.concat([tf.concat([w_c[j+1][1][0], w_c[j][1][0][0]], axis=2),
                                      tf.concat([w_c[j][1][0][1], w_c[j][1][0][2]], axis=2)], axis=1)
            w_c[j][1][1] = tf.concat([tf.concat([w_c[j+1][1][1], w_c[j][1][1][0]], axis=2),
                                      tf.concat([w_c[j][1][1][1], w_c[j][1][1][2]], axis=2)], axis=1)

        w_0 = tf.concat([tf.concat([w_c[j][0][0], w_c[0][0][0][0]], axis=2),
                         tf.concat([w_c[0][0][0][1], w_c[0][0][0][2]], axis=2)], axis=1)
        w_1 = tf.concat([tf.concat([w_c[j][0][1], w_c[0][0][1][0]], axis=2),
                         tf.concat([w_c[0][0][1][1], w_c[0][0][1][2]], axis=2)], axis=1)
        w_2 = tf.concat([tf.concat([w_c[j][1][0], w_c[0][1][0][0]], axis=2),
                         tf.concat([w_c[0][1][0][1], w_c[0][1][0][2]], axis=2)], axis=1)
        w_3 = tf.concat([tf.concat([w_c[j][1][1], w_c[0][1][1][0]], axis=2),
                         tf.concat([w_c[0][1][1][1], w_c[0][1][1][2]], axis=2)], axis=1)

        w_1234 = tf.concat([tf.concat([w_0, w_1], axis=2), tf.concat([w_2, w_3], axis=2)], axis=1)
        return w_1234


class IDTCWT(layers.Layer):
    """
    Inverse Duel Tree Complex Wavelet Transform
    Input: level - tree-level (int)
    """
    def __init__(self, level=1, caoncatenated=True, **kwargs):
        super(IDTCWT, self).__init__(**kwargs)

        if level <= 1:
            level = 1

        self.level = int(level)
        self.conv_type = "SAME"
        self.border_padd = "SYMMETRIC"

        # Faf - First analysis filter - for the first level
        # Fsf - First synthesis filter
        faf, fsf = filters.fs_farras()
        af, sf = filters.duelfilt()

        self.Faf = duel_filter_tf(faf)
        self.Fsf = duel_filter_tf(fsf)
        self.af = duel_filter_tf(af)
        self.sf = duel_filter_tf(sf)

        self.caoncatenated = caoncatenated
    def build(self, input_shape):
        # repeat last channel if input channel bigger then 1
        if input_shape[-1] > 1:
            self.Faf = tf.repeat(self.Faf, input_shape[-1], axis=-1)
            self.Fsf = tf.repeat(self.Fsf, input_shape[-1], axis=-1)
            self.af = tf.repeat(self.af, input_shape[-1], axis=-1)
            self.sf = tf.repeat(self.sf, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        # convert one big image into list of tree levels
        
        if self.caoncatenated:
            w_rec = reconstruct_w_leveln(inputs, self.level)
        else:
            w_rec = inputs

        height = int(w_rec[0][0][0][0].shape[1]*2)
        width = int(w_rec[0][0][0][0].shape[2]*2)

        # init image to be reconstructed
        y = tf.zeros((height, width, inputs.shape[-1]), dtype=tf.float32)

        w_i = [[[[list() for _ in range(3)], [list() for _ in range(3)]]
                for __ in range(2)] for ___ in range(self.level+1)]

        # first add and subtract (inverse the transform)
        for j in range(self.level):
            for m in range(3):

                w_i[j][0][0][m], w_i[j][1][1][m] = add_sub(w_rec[j][0][0][m], w_rec[j][1][1][m])
                w_i[j][0][1][m], w_i[j][1][0][m] = add_sub(w_rec[j][0][1][m], w_rec[j][1][0][m])

        # synthesis with the First filters to be last in the reconstruction
        for m in range(2):
            for n in range(2):
                lo = w_rec[self.level][m][n]
                for j in [x for x in range(1, self.level)][::-1]:
                    lo = synthesis_filter_bank2d(lo, w_i[j][m][n], self.sf[m][0],
                                                 self.sf[m][1], self.sf[n][0], self.sf[n][1])
                lo = synthesis_filter_bank2d(lo, w_i[0][m][n], self.Fsf[m][0],
                                             self.Fsf[m][1], self.Fsf[n][0], self.Fsf[n][1])
                y = tf.math.add(y, lo)

        # revert the normalization
        y = tf.math.divide(y, 2)
        return y


if __name__ == "__main__":
    pass
    # from tensorflow.keras.datasets import mnist, cifar10
    # from tensorflow.keras.optimizers import Adam, SGD
    # from tensorflow.keras.utils import to_categorical

    # img = cv2.imread("../input/Lenna_orig.png", 0)
    # img_ex1 = np.expand_dims(img, axis=0)
    # #
    # if len(img_ex1.shape) <= 3:
    #     img_ex1 = np.expand_dims(img_ex1, axis=-1)
    #
    # _, h, w, c = img_ex1.shape
    # #
    #
    # cplx_input = layers.Input(shape=(h, w, c))
    # x = DTCWT(2)(cplx_input)
    # # x = IDTCWT(2)(x)
    # model = Model(cplx_input, x, name="mymodel")
    # model.summary()
    #
    # out = model.predict(img_ex1)
    # # diff = np.max(out[0] - img)
    # # print("diff is", diff)
    # cv2.imshow("orig", out[0].astype('uint8'))
    # # cv2.imshow("reconstructed", img.astype('uint8'))
    # cv2.waitKey(0)
    # x = layers.Conv2D(32, (3, 3), activation="relu",padding='same')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(10, activation="softmax")(x)
    # model = Model(cplx_input, x, name="mymodel")
    # model.summary()
    #
    # optimizer = SGD(lr=1e-4, momentum=0.9)
    # model.compile(loss="categorical_crossentropy",
    #               optimizer=optimizer, metrics=["accuracy"])
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # x_train = x_train.astype('float32') / 255.0
    # #x_train = np.expand_dims(x_train, axis=-1)
    #
    # x_test = x_test.astype('float32') / 255.0
    # #x_test = np.expand_dims(x_test, axis=-1)
    # print(x_test.shape)
    # history = model.fit(x_train, y_train,
    #                     validation_split=0.2,
    #                     epochs=30,
    #                     batch_size=32,
    #                     verbose=2,
    #                     )

