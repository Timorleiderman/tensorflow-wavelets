% synthesis filter 2d
function y = synthesis_filter_bank2d(cA, cD, LoR, HiR)

[h, w] = size(cA);
L = length(LoR)/2;
L1 = length(LoR);
conv_type = 'same';
% extend transpse for column
LL = wextend('addrow','zpd',cA', L);
LH = wextend('addrow','zpd',cD{1}', L);
% upsample zero insertion
[h_ex, w_ex] = size(LL);
LL_us = zeros(2*h_ex,w_ex);
LH_us = zeros(2*h_ex,w_ex);
LL_us(1:2:end,:) =  LL;
LH_us(1:2:end,:) =  LH;
%apply low pass filter
LL_us_conv = conv2(LL_us,LoR(:),conv_type);
%apply high pass filter
LH_us_conv = conv2(LH_us,HiR(:),conv_type);
% add approximation and details
LL_LH = LL_us_conv + LH_us_conv;
LL_LH = LL_LH(L+1:end-L-2,:);
LL_LH(1:L1-2, :) = LL_LH(1:L1-2, :) + LL_LH(w_ex+[1:L1-2], :);
LL_LH = LL_LH(1:2*w, :);
% circular shift
n = 0:2*w-1;
n = mod(n-(1-L1/2), 2*w);
LL_LH = LL_LH(n+1,:)';

HL = wextend('addrow','zpd',cD{2}', L);
HH = wextend('addrow','zpd',cD{3}', L);
[h_ex, w_ex] = size(HL);
HL_us = zeros(2*h_ex,w_ex);
HH_us = zeros(2*h_ex,w_ex);
HL_us(1:2:end,:) =  HL;
HH_us(1:2:end,:) =  HH;
%apply low pass filter
HL_us_conv = conv2(HL_us,LoR(:),conv_type);
%apply high pass filter
HH_us_conv = conv2(HH_us,HiR(:),conv_type);
% add details and details
HL_HH = HL_us_conv + HH_us_conv;
HL_HH = HL_HH(L+1:end-L-2,:);

HL_HH(1:L1-2, :) = HL_HH(1:L1-2, :) + HL_HH(w_ex+[1:L1-2], :);

% circular shift
n = 0:2*w-1;
n = mod(n-(1-L1/2), 2*w);
HL_HH = HL_HH(n+1,:)';

% same process to the resaults
LL_LH_ex = wextend('addrow','zpd',LL_LH, L);
HL_HH_ex = wextend('addrow','zpd',HL_HH, L);
[h_ex, w_ex] = size(LL_LH_ex);

LL_LH_us = zeros(2*h_ex,w_ex);
HL_HH_us = zeros(2*h_ex,w_ex);
LL_LH_us(1:2:end,:) =  LL_LH_ex;
HL_HH_us(1:2:end,:) =  HL_HH_ex;



LL_LH_us_conv = conv2(LL_LH_us,LoR(:),conv_type);
HL_HH_us_conv = conv2(HL_HH_us,HiR(:),conv_type);



y = LL_LH_us_conv + HL_HH_us_conv;
y = y(L+1:end-L-2,:);

y(1:L1-2, :) = y(1:L1-2, :) + y(2*h+[1:L1-2], :);
y = y(1:2*h, :);
% circular shift
n = 0:2*h-1;
n = mod(n-(1-L1/2), 2*h);
y = y(n+1,:);
% output
lo = sfb2D_A(cA,    cD{1}, [LoR(:), HiR(:)], 2);
hi = sfb2D_A(cD{2}, cD{3}, [LoR(:), HiR(:)], 2);
% filter along columns
y_ref = sfb2D_A(lo, hi, [LoR(:), HiR(:)], 1);






end