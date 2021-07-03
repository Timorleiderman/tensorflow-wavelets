% db2 wavelet decomposition filter size should be 4
% input X vector decomposition filters low pass and high pass

function [cA, cD] = mydwt(X, LoD, HiD)

lf = length(LoD);
% extend the signal by the width of the filter
f = wextend('1d','sym',X, lf-1);

conv_type = 'same';
% convolve and down sample for approximation
a = conv(f,LoD,conv_type);
a_ds = a(1:2:end);
% convolve and down sample for details
d = conv(f,HiD,conv_type);
d_ds = d(1:2:end);

cA = a_ds(lf/2:end-lf/2+1);
cD = d_ds(lf/2:end-lf/2+1);

end
