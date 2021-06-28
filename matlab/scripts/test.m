clear
N = 256;
f = [1:N]';

wavelet_name = 'db2';
[LoD,HiD] = wfilters(wavelet_name,'d'); % decomposition
[LoR,HiR] = wfilters(wavelet_name,'r'); % reconstruction

[cA,cD] = dwt(f,LoD,HiD,'mode','sym');
fidwt = idwt(cA,cD,LoR,HiR,'mode','sym');

% 
% [cA,cD] = dwt(f,wavelet_name,'mode','sym');
% f11 = idwt(cA,cD,wavelet_name,'mode','sym');


% f_ex = wextend('1d','sym',f,1)

a = conv(f,LoD,'full');
a_ds = a(3:2:end-2);
d = conv(f,HiD,'full');
d_ds = d(3:2:end-2);

a_us = zeros(N,1);
d_us = zeros(N,1);

a_us(1:2:end) =  a_ds;
d_us(1:2:end) =  d_ds;

fmy = conv(a_us,LoR,'full') + conv(d_us,HiR,'full');
fmy = fmy(2:end-2);


disp(strcat((['Error |f-fmy|/|f| = ' num2str(norm(f-fmy)/norm(f))])));

disp(strcat((['Error |f-fidwt|/|f| = ' num2str(norm(f-fidwt)/norm(f))])));

