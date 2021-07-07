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
        db2_lpf = tf.reshape(db2_lpf, (1, 4, 1, 1))
        self.db2_lpf = tf.repeat(db2_lpf, 3, axis=-1)

        db2_hpf = tf.constant(db2_hpf)
        db2_hpf = tf.reshape(db2_hpf, (1, 4, 1, 1))
        self.db2_hpf = tf.repeat(db2_hpf, 3, axis=-1)

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


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # x_train = cv2.imread("../input/LennaGrey.png", 0)
    frog = tf.expand_dims(
        x_train[0, :, :, :], 0, name=None
    )
    print("frog shape", frog.shape)
    model = keras.Sequential()
    model.add(keras.Input(shape=(32, 32, 3)))
    model.add(DWT())
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