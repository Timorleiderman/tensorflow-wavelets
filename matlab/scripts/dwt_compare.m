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

sf = analysis_filter(end:-1:1, :);

x = 1:64;
N = length(x);
L = length(analysis_filter)/2;

[ca0, cd0] = mydwt(x, analysis_filter(:,1), analysis_filter(:,2));
y0 = myidwt(ca0, cd0, sf(:,1), sf(:,2))';

[ca1, cd1] = dwt(x, analysis_filter(:,1), analysis_filter(:,2));
y1 = myidwt(ca1, cd1, sf(:,1), sf(:,2))';

x_circ_shift = cshift(x,-L);

% lowpass filter
lo = upfirdn(x_circ_shift, analysis_filter(:,1), 1, 2);
lo(1:L) = lo(N/2+[1:L]) + lo(1:L);
ca2 = lo(1:N/2);

% highpass filter
hi = upfirdn(x_circ_shift, analysis_filter(:,2), 1, 2);
hi(1:L) = hi(N/2+[1:L]) + hi(1:L);
cd2 = hi(1:N/2);

N = 2*length(ca2);
L = length(sf);
lo = upfirdn(ca2, sf(:,1), 2, 1);
hi = upfirdn(cd2, sf(:,2), 2, 1);
y = lo + hi;
y(1:L-2) = y(1:L-2) + y(N+[1:L-2]);
y = y(1:N);
y2 = cshift(y, 1-L/2);

