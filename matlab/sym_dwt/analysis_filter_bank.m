function [cA, cD] = analysis_filter_bank(x, LoD, HiD)

N = length(x);
L = length(LoD)/2;


% shif by L samples to the Right
n = 0:N-1;
n = mod(n+L, N);
x = x(n+1);

% When the analysis and synthesis filters are exactly symmetric
% for symetric filter bank we need to use zero padd 
x_extend = wextend('1d','zpd',x, L);

lo_conv = conv(x_extend, LoD(:),'same');
lo_conv = lo_conv(1:2:end);

% To avoid this excessive length, the last L/2 samples of each subband
% signal is added to the first L/2 samples
% This procedure (periodic extension) can create undesirable 
% artifacts at the beginning and end of the subband signals,
% however, it is the most convenient solution.
% When the analysis and synthesis filters are exactly symmetric,
% a different procedure (symmetric extension) can be used,
% that avoids the artifacts associated with periodic extension.
lo_conv(1:L) = lo_conv(N/2 + [1:L]) + lo_conv(1:L);

hi_conv = conv(x_extend, HiD(:),'same');
hi_conv = hi_conv(1:2:end);
hi_conv(1:L) = hi_conv(N/2+[1:L]) + hi_conv(1:L);

cA = lo_conv(1:N/2);
cD = hi_conv(1:N/2);

end
