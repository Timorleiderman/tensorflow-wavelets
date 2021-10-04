% haar wavelet decomposition 
% input grayscale image
function [LL, LH, HL, HH] = dwt2d_db2(img)

a_j = ( img(:,1:2:end) + img(:,2:2:end) ) * (1/sqrt(2));
d_j = ( img(:,1:2:end) - img(:,2:2:end) ) * (1/sqrt(2));

% second apply horizontal transform on the rows
LL = (a_j(1:2:end,:) + a_j(2:2:end,:)) * (1/sqrt(2)); % LL
LH = (a_j(1:2:end,:) - a_j(2:2:end,:)) * (1/sqrt(2)); % LH
HL = (d_j(1:2:end,:) + d_j(2:2:end,:)) * (1/sqrt(2)); % HL
HH = (d_j(1:2:end,:) - d_j(2:2:end,:)) * (1/sqrt(2)); % HH

end
