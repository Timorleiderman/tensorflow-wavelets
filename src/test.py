import cv2
import unittest

from tensorflow_wavelets.utils.models import *
from tensorflow_wavelets.utils.mse import *
from tensorflow_wavelets.utils.data import *

# install dependencies on venv
# conda install scipy
# conda install matplotlib
# pip install psnr-hvsm
# conda install scikit-image
# conda install -c conda-forge opencv


class TestSrc(unittest.TestCase):

    def test_dwt_idwt_sof_thresh(self):

        img = cv2.imread("../input/LennaGrey.png", 0)
        img_ex1 = np.expand_dims(img, axis=-1)
        img_ex2 = np.expand_dims(img_ex1, axis=0)
        model = basic_dwt_idw(input_shape=img_ex1.shape, wave_name="db2", eagerly=True, soft_theshold=True)
        rec = model.predict(img_ex2)
        rec = rec[0, ..., 0]
        mse_lim = 0.072
        self.assertLess(mse(img, rec), mse_lim, "Should be less then" + str(mse_lim))

    def test_dwt_idwt(self):

        img = cv2.imread("../input/LennaGrey.png", 0)
        img_ex1 = np.expand_dims(img, axis=-1)
        img_ex2 = np.expand_dims(img_ex1, axis=0)
        model = basic_dwt_idw(input_shape=img_ex1.shape, wave_name="db2", eagerly=True, soft_theshold=False)
        rec = model.predict(img_ex2)
        rec = rec[0, ..., 0]
        mse_lim = 1e-3
        self.assertLess(mse(img, rec), mse_lim, "Should be less then" + str(mse_lim))

    def test_basic_train_mnist(self):
        (x_train, y_train), (x_test, y_test) = load_mnist(remove_n_samples=0)

        model = AutocodeBasicDWT(latent_dim=64, width=28, height=28)
        model.compile(optimizer='adam', loss="mse")
        model.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))

        encoded_imgs = model.encoder(x_test).numpy()
        decoded_imgs = model.decoder(encoded_imgs).numpy()
        for img_dec, img_test in zip(decoded_imgs, x_test):
            self.assertLess(mse(img_dec, img_test), 1e2, "wow")


if __name__ == '__main__':
    unittest.main()
