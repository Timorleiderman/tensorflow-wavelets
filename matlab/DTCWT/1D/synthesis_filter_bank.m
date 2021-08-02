function y = synthesis_filter_bank(cA, cD, LoR, HiR)

N = 2*length(cA);
L = length(LoR)/2;
L1 = length(LoR);
conv_type = 'same';

% extend the signal by the width of the filter
cA = wextend('1d','zpd',cA, L);
cD = wextend('1d','zpd',cD, L);

N1 = 2*length(cA);
a_us = zeros(N1,1);
d_us = zeros(N1,1);
a_us(1:2:end) =  cA;
d_us(1:2:end) =  cD;

a_us_conv = conv(a_us,LoR(:),conv_type)';
d_us_conv = conv(d_us,HiR(:),conv_type)';

a_us_conv = a_us_conv(L+1:end-L-2);
d_us_conv = d_us_conv(L+1:end-L-2);

f = a_us_conv + d_us_conv;
f(1:L1-2) = f(1:L1-2) + f(N+[1:L1-2]);
f = f(1:N);

n = 0:N-1;
n = mod(n+L-1, N);
y = f(n+1);

end