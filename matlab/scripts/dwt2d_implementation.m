
clear

X = imread('../../input/Lenna_orig.png');

% eample for bior1.3
wavelet_name = 'bior1.3';
[LoD,HiD] = wfilters(wavelet_name,'d')