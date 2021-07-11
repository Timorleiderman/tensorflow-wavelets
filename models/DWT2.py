import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, cifar10


class DWT(layers.Layer):
    def __init__(self):
        super(DWT, self).__init__()

        db2_h0 = (1+math.sqrt(3))/(4*math.sqrt(2))
        db2_h1 = (3+math.sqrt(3))/(4*math.sqrt(2))
        db2_h2 = (3-math.sqrt(3))/(4*math.sqrt(2))
        db2_h3 = (1-math.sqrt(3))/(4*math.sqrt(2))

        db2_lpf = [db2_h0, db2_h1, db2_h2, db2_h3]
        db2_hpf = [db2_h3, -db2_h2, db2_h1, -db2_h0]

        db2_lpf = tf.constant(db2_lpf)
        self.db2_lpf = tf.reshape(db2_lpf, (1, 4, 1, 1))

        db2_hpf = tf.constant(db2_hpf)
        self.db2_hpf = tf.reshape(db2_hpf, (1, 4, 1, 1))

    def build(self, input_shape):

        if input_shape[-1] == 3:
            self.db2_lpf = tf.repeat(self.db2_lpf, 3, axis=-1)
            self.db2_hpf = tf.repeat(self.db2_hpf, 3, axis=-1)

    def call(self, inputs, training=None, mask=None):

        inputs_padd = tf.pad(inputs, [[0, 0], [0, 0], [3, 3], [0, 0]], "SYMMETRIC")
        a = tf.nn.conv2d(
            inputs_padd, self.db2_lpf, padding='VALID', strides=[1, 1, 1, 1],
        )
        d = tf.nn.conv2d(
            inputs_padd, self.db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
        )
        a_ds = a[:, :, 1:inputs.shape[1]:2, :]
        d_ds = d[:, :, 1:inputs.shape[1]:2, :]
        a_ds_padd = tf.pad(a_ds, [[0, 0], [3, 3], [0, 0], [0, 0]], "SYMMETRIC")
        d_ds_padd = tf.pad(d_ds, [[0, 0], [3, 3], [0, 0], [0, 0]], "SYMMETRIC")

        a_ds_padd = tf.transpose(a_ds_padd, perm=[0, 2, 1, 3])
        d_ds_padd = tf.transpose(d_ds_padd, perm=[0, 2, 1, 3])

        aa = tf.nn.conv2d(
            a_ds_padd, self.db2_lpf, padding='VALID', strides=[1, 1, 1, 1],
        )
        ad = tf.nn.conv2d(
            a_ds_padd, self.db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
        )
        da = tf.nn.conv2d(
            d_ds_padd, self.db2_lpf, padding='VALID', strides=[1, 1, 1, 1],
        )
        dd = tf.nn.conv2d(
            d_ds_padd, self.db2_hpf, padding='VALID', strides=[1, 1, 1, 1],
        )

        aa = tf.transpose(aa, perm=[0, 2, 1, 3])
        ad = tf.transpose(ad, perm=[0, 2, 1, 3])
        da = tf.transpose(da, perm=[0, 2, 1, 3])
        dd = tf.transpose(dd, perm=[0, 2, 1, 3])

        LL = aa[:, 1:inputs.shape[2]:2, :, :]
        LH = ad[:, 1:inputs.shape[2]:2, :, :]
        HL = da[:, 1:inputs.shape[2]:2, :, :]
        HH = dd[:, 1:inputs.shape[2]:2, :, :]

        x = tf.concat([LL, LH, HL, HH], axis=-1)
        return x


class IDWT(layers.Layer):
    def __init__(self):
        super(IDWT, self).__init__()

        self.padd_type = "VALID"
        # calculate Decomposition LPF and HPF
        db2_h0 = (1+math.sqrt(3))/(4*math.sqrt(2))
        db2_h1 = (3+math.sqrt(3))/(4*math.sqrt(2))
        db2_h2 = (3-math.sqrt(3))/(4*math.sqrt(2))
        db2_h3 = (1-math.sqrt(3))/(4*math.sqrt(2))

        db2_lpfR = [db2_h3, db2_h2, db2_h1, db2_h0]
        db2_hpfR = [-db2_h0, db2_h1, -db2_h2, db2_h3]

        # db2_lpf = [db2_h0, db2_h1, db2_h2, db2_h3]
        # db2_hpf = [db2_h3, -db2_h2, db2_h1, -db2_h0]

        db2_lpf = tf.constant(db2_lpfR)
        self.db2_lpf = tf.reshape(db2_lpf, (1, 4, 1, 1))

        db2_hpf = tf.constant(db2_hpfR)
        self.db2_hpf = tf.reshape(db2_hpf, (1, 4, 1, 1))

    def upsampler2d(self, x):
        # zero_tensor = tf.zeros(shape=x.shape, dtype=tf.float32)
        zero_tensor = tf.zeros_like(x, dtype=tf.float32)
        stack_rows = tf.stack([x, zero_tensor], axis=3)
        stack_rows = tf.reshape(stack_rows, shape=[-1, x.shape[1], x.shape[2]*2, x.shape[3]])
        stack_rows = tf.transpose(stack_rows, perm=[0, 2, 1, 3])
        # zero_tensor_1 = tf.zeros(shape=stack_rows.shape, dtype=tf.float32)
        zero_tensor_1 = tf.zeros_like(stack_rows, dtype=tf.float32)
        stack_rows_cols = tf.stack([stack_rows, zero_tensor_1], axis=3)
        us_padded = tf.reshape(stack_rows_cols, shape=[-1, x.shape[1]*2, x.shape[2]*2, x.shape[3]])
        us_padded = tf.transpose(us_padded, perm=[0, 2, 1, 3])
        return us_padded

    def call(self, inputs, training=None, mask=None):
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "SYMMETRIC")
        x = tf.cast(x, tf.float32)

        LL = tf.expand_dims(x[:, :, :, 0], axis=-1)
        LH = tf.expand_dims(x[:, :, :, 1], axis=-1)
        HL = tf.expand_dims(x[:, :, :, 2], axis=-1)
        HH = tf.expand_dims(x[:, :, :, 3], axis=-1)

        print(LL.shape)
        LL_us_pad = self.upsampler2d(LL)
        LH_us_pad = self.upsampler2d(LH)
        HL_us_pad = self.upsampler2d(HL)
        HH_us_pad = self.upsampler2d(HH)

        LL_conv_lpf = tf.nn.conv2d(LL_us_pad, self.db2_lpf, padding=self.padd_type, strides=[1, 1, 1, 1],)
        LL_conv_lpf_tr = tf.transpose(LL_conv_lpf, perm=[0, 2, 1, 3])
        LL_conv_lpf_lpf = tf.nn.conv2d(LL_conv_lpf_tr, self.db2_lpf, padding=self.padd_type, strides=[1, 1, 1, 1],)
        LL_conv_lpf_lpf_tr = tf.transpose(LL_conv_lpf_lpf, perm=[0, 2, 1, 3])

        LH_conv_lpf = tf.nn.conv2d(LH_us_pad, self.db2_lpf, padding=self.padd_type, strides=[1, 1, 1, 1],)
        LH_conv_lpf_tr = tf.transpose(LH_conv_lpf, perm=[0, 2, 1, 3])
        LH_conv_lpf_hpf = tf.nn.conv2d(LH_conv_lpf_tr, self.db2_lpf, padding=self.padd_type, strides=[1, 1, 1, 1],)
        LH_conv_lpf_hpf_tr = tf.transpose(LH_conv_lpf_hpf, perm=[0, 2, 1, 3])

        HL_conv_hpf = tf.nn.conv2d(HL_us_pad, self.db2_hpf, padding=self.padd_type, strides=[1, 1, 1, 1],)
        HL_conv_hpf_tr = tf.transpose(HL_conv_hpf, perm=[0, 2, 1, 3])
        HL_conv_hpf_lpf = tf.nn.conv2d(HL_conv_hpf_tr, self.db2_lpf, padding=self.padd_type, strides=[1, 1, 1, 1],)
        HL_conv_hpf_lpf_tr = tf.transpose(HL_conv_hpf_lpf, perm=[0, 2, 1, 3])

        HH_conv_hpf = tf.nn.conv2d(HH_us_pad, self.db2_hpf, padding=self.padd_type, strides=[1, 1, 1, 1],)
        HH_conv_hpf_tr = tf.transpose(HH_conv_hpf, perm=[0, 2, 1, 3])
        HH_conv_hpf_hpf = tf.nn.conv2d(HH_conv_hpf_tr, self.db2_hpf, padding=self.padd_type, strides=[1, 1, 1, 1],)
        HH_conv_hpf_hpf_tr = tf.transpose(HH_conv_hpf_hpf, perm=[0, 2, 1, 3])

        LL_LH = tf.math.add(LL_conv_lpf_lpf_tr, LH_conv_lpf_hpf_tr)
        HL_HH = tf.math.add(HL_conv_hpf_lpf_tr, HH_conv_hpf_hpf_tr)

        reconstructed = tf.math.add(LL_LH, HL_HH)
        return reconstructed[:, 5:-4, 5:-4, :]



if __name__ == "__main__":
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = x_train.astype("float32")
    # x_test = x_test.astype("float32")
    # # x_train = cv2.imread("../input/LennaGrey.png", 0)
    # frog = tf.expand_dims(
    #     x_train[0, :, :, :], 0, name=None
    # )
    # print("frog shape", frog.shape)
    model = keras.Sequential()
    model.add(keras.Input(shape=(256, 256, 4)))
    model.add(IDWT())
    model.summary()



    # a = model.predict(frog, steps=1)
    # #
    # approx = tf.image.convert_image_dtype(a[0, ..., 0], dtype=tf.float32)
    # with tf.Session() as sess:
    #     img = sess.run(approx)
    # #     pass
    # #
    # img = np.clip(img, 0, 255)
    # img = np.ceil(img)
    # img = img.astype("uint8")
    # with open(r"D:\TEMP\LL_python_layer.raw", "wb") as outfile:
    #     outfile.write(img)  # Write it

    # model = models.WaveletCifar10CNN.WaveletCNN((32,32,3), 10)
    # model.summary()