% db2 wavelet reconstruction 
% input aprroximation and detail coeffitients
% reconstruction filters low pass and high pass filters

function [X] = myidwt(cA, cD, LoR, HiR)

lr = length(LoR);

conv_type = 'valid';

% extend the signal by the width of the filter
cA = wextend('1d','sym',cA, lr-1);
cD = wextend('1d','sym',cD, lr-1);

N = length(cA)*2;
%upsample
a_us = zeros(N,1);
d_us = zeros(N,1);
a_us(1:2:end) =  cA;
d_us(1:2:end) =  cD;

% convolve
f = conv(a_us,LoR,conv_type) + conv(d_us,HiR,conv_type);
%output without borders
f = f(lr*3-2:end-lr*2-2);

X = f;



end
