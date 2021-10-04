
import numpy as np
import cv2
from tensorflow_wavelets.utils.mse import mse
import pywt

img = cv2.imread("../input/LennaGrey.png", 0)


# print(threshold)

coeffs2 = pywt.dwt2(img, 'db2')

sigma = np.median(np.abs(coeffs2[1][2])) / 0.67448975

threshold = sigma**2 / np.sqrt(max(coeffs2[1][2].var()**2 - sigma**2, 0))

print(threshold)
ret, thresh4 = cv2.threshold(coeffs2[1][2], threshold, 255, cv2.THRESH_TOZERO)

coeffs_thresh = [coeffs2[0], (coeffs2[1][0], coeffs2[1][1], thresh4)]

rec = pywt.idwt2(coeffs_thresh, 'db2')
# rec = pywt.idwt2(coeffs2, 'db2')


print("Hey Wavelets")
print(mse(rec, img))
cv2.imshow("img", rec.astype('uint8'))
cv2.waitKey()