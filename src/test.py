import cv2
import unittest
import os

from tensorflow_wavelets.utils.models import *
from tensorflow_wavelets.utils.mse import *
from tensorflow_wavelets.utils.data import *

# install dependencies on venv
# conda install scipy
# conda install matplotlib
# pip install psnr-hvsm
# conda install scikit-image
# conda install -c conda-forge opencv


# unitests for Development 
class TestSrc(unittest.TestCase):
    '''
        run from src dir.
        if not change the path of lenna_input_path
    '''
    lenna_input_path = "../Development/input/LennaGrey.png"
    
    def test_dwt_idwt_sof_thresh(self):

        img = cv2.imread(self.lenna_input_path, 0)
        self.assertIsNotNone(img, "LennaGrey.png not found in " + self.lenna_input_path)
        img_ex1 = np.expand_dims(img, axis=-1)
        img_ex2 = np.expand_dims(img_ex1, axis=0)
        model = basic_dwt_idwt(input_shape=img_ex1.shape, wave_name="db2", eagerly=True, theshold=True, mode='soft', algo='sure')
        rec = model.predict(img_ex2)
        rec = rec[0, ..., 0]
        mse_lim = 3.5
        self.assertLess(mse(img, rec), mse_lim, "Should be less than " + str(mse_lim))

    def test_dwt_idwt_hard_thresh(self):

        img = cv2.imread(self.lenna_input_path, 0)
        self.assertIsNotNone(img, "LennaGrey.png not found in " + self.lenna_input_path)
        img_ex1 = np.expand_dims(img, axis=-1)
        img_ex2 = np.expand_dims(img_ex1, axis=0)
        model = basic_dwt_idwt(input_shape=img_ex1.shape, wave_name="db2", eagerly=True, theshold=True, mode='hard', algo='sure')
        rec = model.predict(img_ex2)
        rec = rec[0, ..., 0]
        mse_lim = 3.5
        self.assertLess(mse(img, rec), mse_lim, "Should be less than " + str(mse_lim))

    def test_dwt_idwt(self):

        img = cv2.imread(self.lenna_input_path, 0)
        self.assertIsNotNone(img, "LennaGrey.png not found in " + self.lenna_input_path)
        img_ex1 = np.expand_dims(img, axis=-1)
        img_ex2 = np.expand_dims(img_ex1, axis=0)
        model = basic_dwt_idwt(input_shape=img_ex1.shape, wave_name="db2", eagerly=True, theshold=False)
        rec = model.predict(img_ex2)
        rec = rec[0, ..., 0]
        mse_lim = 1e-3
        self.assertLess(mse(img, rec), mse_lim, "Should be less than " + str(mse_lim))

    def test_dwt_idwt_not_concat(self):

        img = cv2.imread(self.lenna_input_path, 0)
        self.assertIsNotNone(img, "LennaGrey.png not found in " + self.lenna_input_path)
        img_ex1 = np.expand_dims(img, axis=-1)
        img_ex2 = np.expand_dims(img_ex1, axis=0)
        model = basic_dwt_idwt(input_shape=img_ex1.shape, wave_name="db2", eagerly=True, theshold=False, concat = False)
        rec = model.predict(img_ex2)
        rec = rec[0, ..., 0]
        mse_lim = 1e-3
        self.assertLess(mse(img, rec), mse_lim, "Should be less than " + str(mse_lim))

    def test_basic_train_mnist(self):
        (x_train, y_train), (x_test, y_test) = load_mnist(remove_n_samples=1000)

        model = AutocodeBasicDWT(latent_dim=64, width=28, height=28)
        model.compile(optimizer='adam', loss="mse")
        model.fit(x_train, x_train, epochs=1, shuffle=True, validation_data=(x_test, x_test), verbose=1)

        encoded_imgs = model.encoder(x_test).numpy()
        decoded_imgs = model.decoder(encoded_imgs).numpy()
        for img_dec, img_test in zip(decoded_imgs, x_test):
            self.assertLess(mse(img_dec, img_test), 1e2, "mse should be less then 0.01")

    def test_dmwt(self):
        (x_train, y_train), (x_test, y_test) = load_mnist(remove_n_samples=1000)
        input_shape = (28, 28, 1)
        model = basic_dmwt(input_shape=input_shape, nb_classes=10, wave_name="ghm", eagerly=True)
        model.compile(loss="categorical_crossentropy",optimizer='adam', metrics=["accuracy"])
        model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1,)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        self.assertGreater(test_acc, 0.8, "test accuracy should be higher then 0.8")
        self.assertLess(test_loss, 0.8, "test loss should be less then 0.8")

    def test_dtcwt(self):
        (x_train, y_train), (x_test, y_test) = load_mnist(remove_n_samples=1000)
        input_shape = (28, 28, 1)
        model = basic_dtcwt(input_shape=input_shape, nb_classes=10, level=2, eagerly=True)
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1,)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        self.assertGreater(test_acc, 0.8, "test accuracy should be higher then 0.8")
        self.assertLess(test_loss, 0.8, "test loss should be less then 0.8")


if __name__ == '__main__':
    unittest.main()
