%  apply filter on one axis
%  for the second axis go again with transpose
function [cA, cD] = analysis_filter_bank2d(x, LoD, HiD)

[h, w] = size(x);
L = length(LoD)/2;

% circular shift 2d
n = 0:h-1;
n = mod(n+L, h);
x = x(n+1,:);

% zero pad borders
x_extend = wextend('addrow','zpd',x, L);

% low pass filter
lo_conv = conv2(x_extend, LoD(:),'same');

% down sample
lo_conv = lo_conv(1:2:end,:);

% circular shift fix
lo_conv(1:L, :) = lo_conv(1:L, :) + lo_conv([1:L]+h/2, :);

% crop
lo_conv = lo_conv(1:h/2, :);

% high pass filter
hi_conv = conv2(x_extend, HiD(:),'same');

% down sample
hi_conv = hi_conv(1:2:end,:);

% circular shift fix
hi_conv(1:L, :) = hi_conv(1:L, :) + hi_conv([1:L]+h/2, :);

% crop
hi_conv = hi_conv(1:h/2, :);

% output
cA = lo_conv;
cD = hi_conv;

end
