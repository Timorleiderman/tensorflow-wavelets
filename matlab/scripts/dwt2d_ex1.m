clear

X = imread('../../input/Lenna_orig.png');
%X_grey = rgb2gray(X)

wavelet_name = 'bior1.3';

[LoD,HiD] = wfilters(wavelet_name,'d')

[cA,cH,cV,cD] = dwt2(X,LoD,HiD,'mode','symh');

X_rec = uint8(idwt2(cA,cH,cV,cD,wavelet_name));
figure(1)
subplot(2,2,1),imshow(uint8(cA))
subplot(2,2,2),imshow(uint8(cH))
subplot(2,2,3),imshow(uint8(cV))
subplot(2,2,4),imshow(uint8(cD))
err = max(max(X - X_rec));