close all
clear 


x = 1:64; % input signal

% analysis filter
af = [
                  0  -0.01122679215254
                  0   0.01122679215254
  -0.08838834764832   0.08838834764832
   0.08838834764832   0.08838834764832
   0.69587998903400  -0.69587998903400
   0.69587998903400   0.69587998903400
   0.08838834764832  -0.08838834764832
  -0.08838834764832  -0.08838834764832
   0.01122679215254                  0
   0.01122679215254                  0
];

sf = af(end:-1:1, :);

N = length(x);
L = length(af)/2;
x = cshift(x,-L);

% lowpass filter
lo = upfirdn(x, af(:,1), 1, 2);
lo(1:L) = lo(N/2+[1:L]) + lo(1:L);
lo = lo(1:N/2);
hi = upfirdn(x, af(:,2), 1, 2);
hi(1:L) = hi(N/2+[1:L]) + hi(1:L);
hi = hi(1:N/2);

% When the analysis and synthesis filters are exactly symmetric
% for symetric filter bank we need to use zero padd 
x_extend = wextend('1d','zpd',x, L);
lo_conv = conv(x_extend, af(:,1),'same');
lo_conv = lo_conv(1:2:end);
lo_conv(1:L) = lo_conv(N/2+[1:L]) + lo_conv(1:L);
lo_conv = lo_conv(1:N/2);


err_lo = max(lo -lo_conv);
% highpass filter


hi_conv = conv(x_extend, af(:,2),'same');
hi_conv = hi_conv(1:2:end);
hi_conv(1:L) = hi_conv(N/2+[1:L]) + hi_conv(1:L);
hi_conv = hi_conv(1:N/2);

err_hi = max(hi - hi_conv);

N = 2*length(lo);
L = length(sf);
lo = upfirdn(lo, sf(:,1), 2, 1);
hi = upfirdn(hi, sf(:,2), 2, 1);
y = lo + hi;
y(1:L-2) = y(1:L-2) + y(N+[1:L-2]);
y = y(1:N);
y = cshift(y, 1-L/2);


L = length(sf(:,1))/2;
conv_type = 'same';
% extend the signal by the width of the filter
cA = wextend('1d','zpd',lo_conv, L);
cD = wextend('1d','zpd',hi_conv, L);

N = 2*length(cA);
a_us = zeros(N,1);
d_us = zeros(N,1);
a_us(1:2:end) =  cA;
d_us(1:2:end) =  cD;

lo_my = conv(a_us,sf(:,1),conv_type)';
hi_my = conv(d_us,sf(:,2),conv_type)';

lo_my = lo_my(L+1:end-L-2);
hi_my = hi_my(L+1:end-L-2);

f = lo_my + hi_my;

N = length(lo_my)-8;
f(1:L-2) = f(1:L-2) + f(N+[1:L-2]);
f = f(1:N);
f = cshift(f, -3);

err = max(y-f);
