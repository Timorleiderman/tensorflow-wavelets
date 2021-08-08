import cv2
import numpy as np
from tensorflow.keras import layers, Model
from utils import filters
from utils.helpers import *
from utils.cast import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical


class DTCWT(layers.Layer):
    def __init__(self, level=1, **kwargs):
        super(DTCWT, self).__init__(**kwargs)

        if level <= 1:
            level = 1

        self.level = level
        self.conv_type = "SAME"
        self.border_padd = "SYMMETRIC"

        Faf, Fsf = filters.FSfarras()
        af, sf = filters.duelfilt()

        self.Faf = duel_filter_tf(Faf)
        self.Fsf = duel_filter_tf(Fsf)
        self.af = duel_filter_tf(af)
        self.sf = duel_filter_tf(sf)

    def build(self, input_shape):
        if input_shape[-1] != 1:
            self.Faf = tf.repeat(self.Faf, input_shape[-1], axis=-1)
            self.Fsf = tf.repeat(self.Fsf, input_shape[-1], axis=-1)
            self.af = tf.repeat(self.af, input_shape[-1], axis=-1)
            self.sf = tf.repeat(self.sf, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        # normalizetion
        x_norm = tf.math.divide(inputs, 2)

        # 2 trees J+1 lists
        w = [[[[], []] for x in range(2)] for i in range(self.level+1)]

        for m in range(2):
            for n in range(2):
                [lo, w[0][m][n]] = analysis_filter_bank2d(x_norm, self.Faf[m][0], self.Faf[m][1], self.Faf[n][0], self.Faf[n][1])
                for j in range(1, self.level):
                    [lo, w[j][m][n]] = analysis_filter_bank2d(lo, self.af[m][0], self.af[m][1],self.af[n][0], self.af[n][1])
                w[self.level][m][n] = lo

        for j in range(self.level):
            for m in range(3):

                w[j][0][0][m], w[j][1][1][m] = add_sub(w[j][0][0][m], w[j][1][1][m])
                w[j][0][1][m], w[j][1][0][m] = add_sub(w[j][0][1][m], w[j][1][0][m])

        # concat into one big image
        j = 1
        w_c = w

        for j in [x for x in range(1, self.level)][::-1]:

            w_c[j][0][0] = tf.concat([tf.concat([w_c[j+1][0][0], w_c[j][0][0][0]], axis=2), tf.concat([w_c[j][0][0][1], w_c[j][0][0][2]], axis=2)], axis=1)
            w_c[j][0][1] = tf.concat([tf.concat([w_c[j+1][0][1], w_c[j][0][1][0]], axis=2), tf.concat([w_c[j][0][1][1], w_c[j][0][1][2]], axis=2)], axis=1)
            w_c[j][1][0] = tf.concat([tf.concat([w_c[j+1][1][0], w_c[j][1][0][0]], axis=2), tf.concat([w_c[j][1][0][1], w_c[j][1][0][2]], axis=2)], axis=1)
            w_c[j][1][1] = tf.concat([tf.concat([w_c[j+1][1][1], w_c[j][1][1][0]], axis=2), tf.concat([w_c[j][1][1][1], w_c[j][1][1][2]], axis=2)], axis=1)


        w_0 = tf.concat([tf.concat([w_c[j][0][0], w_c[0][0][0][0]], axis=2), tf.concat([w_c[0][0][0][1], w_c[0][0][0][2]], axis=2)], axis=1)
        w_1 = tf.concat([tf.concat([w_c[j][0][1], w_c[0][0][1][0]], axis=2), tf.concat([w_c[0][0][1][1], w_c[0][0][1][2]], axis=2)], axis=1)
        w_2 = tf.concat([tf.concat([w_c[j][1][0], w_c[0][1][0][0]], axis=2), tf.concat([w_c[0][1][0][1], w_c[0][1][0][2]], axis=2)], axis=1)
        w_3 = tf.concat([tf.concat([w_c[j][1][1], w_c[0][1][1][0]], axis=2), tf.concat([w_c[0][1][1][1], w_c[0][1][1][2]], axis=2)], axis=1)

        w_1234 = tf.concat([tf.concat([w_0, w_1], axis=2), tf.concat([w_2, w_3], axis=2)], axis=1)
        return w_1234


class IDTCWT(layers.Layer):
    def __init__(self, level=1, **kwargs):
        super(IDTCWT, self).__init__(**kwargs)

        if level <= 1:
            level = 1

        self.level = level
        self.conv_type = "SAME"
        self.border_padd = "SYMMETRIC"

        Faf, Fsf = filters.FSfarras()
        af, sf = filters.duelfilt()

        self.Faf = duel_filter_tf(Faf)
        self.Fsf = duel_filter_tf(Fsf)
        self.af = duel_filter_tf(af)
        self.sf = duel_filter_tf(sf)

    def build(self, input_shape):

        if input_shape[-1] != 1:
            self.Faf = tf.repeat(self.Faf, input_shape[-1], axis=-1)
            self.Fsf = tf.repeat(self.Fsf, input_shape[-1], axis=-1)
            self.af = tf.repeat(self.af, input_shape[-1], axis=-1)
            self.sf = tf.repeat(self.sf, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        w_rec = reconstruct_w_level2(inputs)

        height = int(w_rec[0][0][0][0].shape[1]*2)
        width = int(w_rec[0][0][0][0].shape[2]*2)

        y = tf.zeros((height, width, inputs.shape[-1]), dtype=tf.float32)

        w_i = [[[[list() for x in range(3)], [list() for x in range(3)]] for x in range(2)] for i in range(self.level+1)]

        for j in range(self.level):
            for m in range(3):

                w_i[j][0][0][m], w_i[j][1][1][m] = add_sub(w_rec[j][0][0][m], w_rec[j][1][1][m])
                w_i[j][0][1][m], w_i[j][1][0][m] = add_sub(w_rec[j][0][1][m], w_rec[j][1][0][m])

        for m in range(2):
            for n in range(2):
                lo = w_rec[self.level][m][n]
                for j in [x for x in range(1, self.level)][::-1]:
                    lo = synthesis_filter_bank2d(lo, w_i[j][m][n], self.sf[m][0], self.sf[m][1], self.sf[n][0], self.sf[n][1])
                lo = synthesis_filter_bank2d(lo, w_i[0][m][n], self.Fsf[m][0], self.Fsf[m][1], self.Fsf[n][0], self.Fsf[n][1])
                y = tf.math.add(y, lo)

        y = tf.math.divide(y, 2)
        return y


if __name__ == "__main__":
    img = cv2.imread("../input/Lenna_orig.png", 1)
    img_ex1 = np.expand_dims(img, axis=0)
    #
    if len(img_ex1.shape) <= 3:
        img_ex1 = np.expand_dims(img_ex1, axis=-1)

    _, h, w, c = img_ex1.shape
    #

    cplx_input = layers.Input(shape=(h, w, c))
    x = DTCWT(2)(cplx_input)
    x = IDTCWT(2)(x)
    model = Model(cplx_input, x, name="mymodel")
    model.summary()

    out = model.predict(img_ex1)
    diff = np.max(out[0] - img)
    print("diff is", diff)
    cv2.imshow("orig", out[0].astype('uint8'))
    cv2.imshow("reconstructed", img.astype('uint8'))
    cv2.waitKey(0)
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

