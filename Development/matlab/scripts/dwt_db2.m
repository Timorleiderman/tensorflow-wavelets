% db2 wavelet decomposition 
% input X vector
function [cA, cD] = dwt_db2(X, LoD, HiD)

downsample_X = X(1:2:end);
cA = conv(downsample_X,LoD,'valid');
cD = conv(downsample_X,HiD,'valid');

% s1 = X(1:2:N - 1) + sqrt(3) * X(2:2:N);
% d1 = X(2:2:N) - sqrt(3) / 4 * s1 - (sqrt(3) - 2) / 4 * [s1(1:N / 2); s1(1:N / 2 - 1)];
% s2 = s1 - [d1(2:N / 2); d1(1)];
% cA = (sqrt(3) - 1) / sqrt(2) * s2;
% cD = - (sqrt(3) + 1) / sqrt(2) * d1;


end
