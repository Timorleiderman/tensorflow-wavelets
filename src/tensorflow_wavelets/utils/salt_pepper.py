import numpy as np
import cv2


def addsalt_pepper(img, SNR):

    img_ = img.copy()
    w, h, c = img_.shape
    mask = np.random.choice((0, 1, 2), size=(w, h, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    # Copy by channel to have the same shape as img
    Mask = np.repeat(mask, c, axis=2)
    # salt noise
    img_[Mask == 1] = 255
    # pepper noise
    img_[Mask == 2] = 0
    return img_


if __name__ == "__main__":
    img = cv2.imread("../input/Lenna_orig.png")
    cv2.imwrite( "../input/Lenna_salt_pepper.png", addsalt_pepper(img, 0.9))