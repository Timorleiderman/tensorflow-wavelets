from math import log10, sqrt
import cv2
import numpy as np
from psnr_hvsm import psnr_hvs_hvsm
from skimage import feature

def psnr_hvsm_e(ref, img):

    # convert to yuv color space and pass luma
    orig_yuv = cv2.cvtColor(ref, cv2.COLOR_RGB2YUV)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(orig_yuv[:,:,0].astype('float64')/255, img_yuv[:,:,0].astype('float64')/255)

    return psnr_hvs, psnr_hvsm


def psnr_ed(ref, img):
    ref = ref.astype('float64')
    img = img.astype('float64')

    ref_edge_r = feature.canny(ref[:, :, 0], sigma=3)
    ref_edge_g = feature.canny(ref[:, :, 1], sigma=3)
    ref_edge_b = feature.canny(ref[:, :, 2], sigma=3)

    img_edge_r = feature.canny(img[:, :, 0], sigma=3)
    img_edge_g = feature.canny(img[:, :, 0], sigma=3)
    img_edge_b = feature.canny(img[:, :, 0], sigma=3)

    ed_r = np.mean((ref_edge_r - img_edge_r) ** 2)
    ed_g = np.mean((ref_edge_g - img_edge_g) ** 2)
    ed_b = np.mean((ref_edge_b - img_edge_b) ** 2)

    mse = (ed_r+ed_g+ed_b)/3

    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0

    psnr_ed = 20 * log10(max_pixel / sqrt(mse))

    return psnr_ed

def psnr_e(ref, img):
    ref = ref.astype('float64')
    img = img.astype('float64')

    M = ref.shape[0]
    N = ref.shape[1]

    er = (1/(M*N)) * np.sum((ref[:, :, 0] - img[:, :, 0]) ** 2)
    eg = (1/(M*N)) * np.sum((ref[:, :, 1] - img[:, :, 1]) ** 2)
    eb = (1/(M*N)) * np.sum((ref[:, :, 2] - img[:, :, 2]) ** 2)

    mse = (er + eg + eb)/3

    max_pixel = 255.0
    psnr_e = 10 * log10(max_pixel**2/mse)
    return psnr_e


def psnr_s(ref, img):
    ref = ref.astype('float64')
    img = img.astype('float64')

    w, h, c = ref.shape

    # dividing the image in equal size and non overlapping square regions
    n = 8
    m = 8

    ref_blocks = np.array([ref[i:i + n, j:j + n] for j in range(0, w, n) for i in range(0, h, m)])
    img_blocks = np.array([img[i:i + n, j:j + n] for j in range(0, w, n) for i in range(0, h, m)])

    X = list()
    for ref_block, img_block in zip(ref_blocks, img_blocks):
        Xa = 0.5*(np.mean(ref_block) - np.mean(img_block))**2
        Xp = 0.25*(np.max(ref_block) - np.max(img_block))**2
        Xb = 0.25*(np.min(ref_block) - np.min(img_block))**2
        X.append(Xa+Xp+Xb)

    mse = np.mean(X)

    max_pixel = 255.0
    psnr_s = 10 * log10(max_pixel**2/mse)
    return psnr_s


def psnr(ref, img):
    ref = ref.astype('float64')
    img = img.astype('float64')

    mse = np.mean((ref - img) ** 2)
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

    value = psnr_s(original, compressed_10)
    print(f"PSNR_s value compressed 10%  quality is {value} dB")

    value = psnr_ed(original, compressed_10)
    print(f"PSNR_ed value compressed 10%  quality is {value} dB")


    value = psnr(original, compressed_10)
    print(f"PSNR value compressed 10%  quality is {value} dB")


    print(psnr_hvsm_e(original, compressed_10))
    # value = psnr(original, compressed_100)
    # print(f"PSNR value compressed 100% quality is {value} dB")


if __name__ == "__main__":
    main()