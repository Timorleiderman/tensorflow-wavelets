from math import log10, sqrt
import cv2
import numpy as np


def psnr_e(original, compressed):
    original = original.astype('float64')
    compressed = compressed.astype('float64')

    M = original.shape[0]
    N = original.shape[1]

    er = (1/(M*N)) * np.sum((original[:, :, 0] - compressed[:, :, 0])**2)
    eg = (1/(M*N)) * np.sum((original[:, :, 1] - compressed[:, :, 1])**2)
    eb = (1/(M*N)) * np.sum((original[:, :, 2] - compressed[:, :, 2])**2)

    psnr_e = 10 * log10((er + eg + eb)/3)
    return psnr_e


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    original = cv2.imread("../input/Lenna_orig.png")
    compressed_10 = cv2.imread("../input/Lenna_comp_10.jpg", 1)
    compressed_100 = cv2.imread("../input/Lenna_comp_100.jpg", 1)

    value = psnr_e(original, compressed_10)
    print(f"PSNR_e value compressed 10%  quality is {value} dB")

    value = psnr(original, compressed_10)
    print(f"PSNR value compressed 10%  quality is {value} dB")
    value = psnr(original, compressed_100)
    print(f"PSNR value compressed 100% quality is {value} dB")


if __name__ == "__main__":
    main()