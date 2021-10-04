import pywt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt


class MyHaarFilterBank(object):
    @property
    def filter_bank(self):
        return ([sqrt(2)/2, sqrt(2)/2], [-sqrt(2)/2, sqrt(2)/2],
                [sqrt(2)/2, sqrt(2)/2], [sqrt(2)/2, -sqrt(2)/2])


my_wavelet = pywt.Wavelet('My Haar Wavelet', filter_bank=MyHaarFilterBank())


def print_array(arr):
    print("[%s]" % ", ".join(["%.14f" % x for x in arr]))

# print (pywt.families())
# for family in pywt.families():
#     print("%s family: " % family + ', '.join(pywt.wavelist(family)))

# w = pywt.Wavelet('db3')
# w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)
# print(w.name)
# print(w.short_family_name)
# print(w.family_name)
# print(int(w.dec_len))
# print(int(w.rec_len))

# x = [3, 7, 1, 1, -2, 5, 4, 9]
# cA, cD = pywt.dwt(x, 'haar')
# y = pywt.idwt(cA, cD, 'haar')
#
# print(x)
# print(cA)
# print(cD)
# print(y)



# Load image
original = cv2.imread("../input/Lenna_orig.png", cv2.IMREAD_GRAYSCALE)
# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'Haar')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()