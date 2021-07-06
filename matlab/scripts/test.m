clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N = 256;
% f = [1:N]';
% 
% wavelet_name = 'db2';
% [LoD,HiD] = wfilters(wavelet_name,'d'); % decomposition
% [LoR,HiR] = wfilters(wavelet_name,'r'); % reconstruction
% 
% [cA, cD] = dwt(f,LoD,HiD,'mode','sym');
% [cAmy, cDmy] = mydwt(f,LoD,HiD);
% 
% fidwt = idwt(cA,cD,LoR,HiR,'mode','sym');
% fmyidwt = myidwt(cA,cD,LoR,HiR);
% 
% disperr(cA, cAmy)
% disperr(fidwt, fmyidwt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = imread('../../input/LennaGrey.png');
% f = double(rgb2gray(f));
wavelet_name = 'db2';
[LoD,HiD] = wfilters(wavelet_name,'d'); % decomposition
[LoR,HiR] = wfilters(wavelet_name,'r'); % reconstruction

[cA,cH,cV,cD] = dwt2(f,LoD,HiD,'mode','sym');

[cAmy,cHmy,cVmy,cDmy] = mydwt2d(f,LoD,HiD);
% 
% figure(1)
% imshow(uint8(cA))
% figure(2)
% imshow(uint8(cAmy))
% % 
% [cA,cD] = dwt(f,wavelet_name,'mode','sym');
% f11 = idwt(cA,cD,wavelet_name,'mode','sym');

% border_pad = 4;
% f = wextend('1d','sym',f,border_pad)
% conv_type = 'same';
% a = conv(f,LoD,conv_type);
% a_ds = a(1:2:end);
% d = conv(f,HiD,conv_type);
% d_ds = d(1:2:end);
% a_us = zeros(N+border_pad*2,1);
% d_us = zeros(N+border_pad*2,1);
% a_us(1:2:end) =  a_ds;
% d_us(1:2:end) =  d_ds;
% fmy = conv(a_us,LoR,conv_type) + conv(d_us,HiR,conv_type);
% fmy = fmy(border_pad:end-border_pad-1);
% f = f(border_pad+1:end-border_pad);
% 
% disp(strcat((['Error |f-fmy|/|f| = ' num2str(norm(f-fmy)/norm(f))])));
% 
% disp(strcat((['Error |f-fidwt|/|f| = ' num2str(norm(f-fidwt)/norm(f))])));
% 
