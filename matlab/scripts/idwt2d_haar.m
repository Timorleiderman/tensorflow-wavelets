% haar wavelet decomposition 
% input grayscale image
function img = idwt2d_haar(LL, LH, HL, HH)

[w, h] = size(LL);
a_jj = zeros(w*2,h);
d_jj = zeros(w*2,h);

img = zeros(w*2,h*2);

a_jj(1:2:end,:) = (LL + LH) .*  (1/sqrt(2)); 
a_jj(2:2:end,:) = (LL - LH) .*  (1/sqrt(2)); 
d_jj(1:2:end,:) = (HL + HH) .*  (1/sqrt(2)); 
d_jj(2:2:end,:) = (HL - HH) .*  (1/sqrt(2)); 

img(:,1:2:end) = ( a_jj + d_jj ) * (1/sqrt(2));
img(:,2:2:end) = ( a_jj - d_jj ) * (1/sqrt(2));


end
