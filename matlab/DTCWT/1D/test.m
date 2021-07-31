close all
clear

analysis_filter = [
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
x = 1:64;
N = length(x);
L = length(analysis_filter)/2;

x_circ_shift = cshift(x,-L);

% lowpass filter
lo = upfirdn(x_circ_shift, analysis_filter(:,1), 1, 2);
lo(1:L) = lo(N/2+[1:L]) + lo(1:L);
lo = lo(1:N/2);

% highpass filter
hi = upfirdn(x_circ_shift, analysis_filter(:,2), 1, 2);
hi(1:L) = hi(N/2+[1:L]) + hi(1:L);
hi = hi(1:N/2);

x_extend = wextend('1d','sym',x, L-1);
conv_type = 'valid';
% convolve and down sample for approximation
a = conv(x_extend,analysis_filter(:,1),conv_type);
a_ds = a(1:2:end);

% convolve and down sample for details
d = conv(x_extend,analysis_filter(:,2),conv_type);
d_ds = d(1:2:end);

