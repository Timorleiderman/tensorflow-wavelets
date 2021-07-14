clear

f = imread('../../input/LennaGrey.png');
% f = double(rgb2gray(f));
wavelet_name = 'haar';

[LoD,HiD] = wfilters(wavelet_name,'d'); % decomposition
[LoR,HiR] = wfilters(wavelet_name,'r'); % reconstruction

[cA,cH,cV,cD] = dwt2(f,LoD,HiD,'mode','sym');
reconstructed = idwt2(cA,cH,cV,cD,LoR,HiR);

[cAmy,cHmy,cVmy,cDmy] = mydwt2d(f,LoD,HiD);
myreconstructed = myidwt2d(cAmy,cHmy,cVmy,cDmy,LoR,HiR);

disperr2d(myreconstructed,reconstructed)

figure(1)
imshow(uint8(myreconstructed))
figure(2)
imshow(uint8(reconstructed))
