
function [cA, cH, cV, cD] = analysis_filter_bank2d(x, LoD_row, HiD_row, LoD_col,HiD_col)

if nargin < 4
   LoD_col = LoD_row;
   HiD_col = HiD_row;
end

[h, w] = size(x);
L = length(LoD_row)/2;
conv_type = 'same';
% circular shift 2d
n = 0:h-1;
n = mod(n+L, h);
x = x(n+1,:);

% zero pad borders
x_extend = wextend('addrow','zpd',x, L);

% low pass filter
lo_conv = conv2(x_extend, LoD_row(:),conv_type);

% down sample
lo_conv = lo_conv(1:2:end,:);

% circular shift fix
lo_conv(1:L, :) = lo_conv(1:L, :) + lo_conv([1:L]+h/2, :);

% crop
lo_conv = lo_conv(1:h/2, :);

% high pass filter
hi_conv = conv2(x_extend, HiD_row(:),conv_type);

% down sample
hi_conv = hi_conv(1:2:end,:);

% circular shift fix
hi_conv(1:L, :) = hi_conv(1:L, :) + hi_conv([1:L]+h/2, :);

% crop
hi_conv = hi_conv(1:h/2, :);

lo_conv = lo_conv';
hi_conv = hi_conv';

[h, w] = size(lo_conv);
n = 0:h-1;
n = mod(n+L, h);
% circular shift
lo_conv_cs = lo_conv(n+1,:);
hi_conv_cs = hi_conv(n+1,:);

lo_ext = wextend('addrow','zpd',lo_conv_cs, L);
hi_ext = wextend('addrow','zpd',hi_conv_cs, L);

% low pass filter
lo_lo = conv2(lo_ext, LoD_col(:),conv_type);
lo_hi = conv2(lo_ext, HiD_col(:),conv_type);
hi_lo = conv2(hi_ext, LoD_col(:),conv_type);
hi_hi = conv2(hi_ext, HiD_col(:),conv_type);

% down sample
lo_lo_ds = lo_lo(1:2:end,:);
lo_hi_ds = lo_hi(1:2:end,:);
hi_lo_ds = hi_lo(1:2:end,:);
hi_hi_ds = hi_hi(1:2:end,:);

% circular shift fix
lo_lo_ds(1:L, :) = lo_lo_ds(1:L, :) + lo_lo_ds([1:L]+h/2, :);
lo_hi_ds(1:L, :) = lo_hi_ds(1:L, :) + lo_hi_ds([1:L]+h/2, :);
hi_lo_ds(1:L, :) = hi_lo_ds(1:L, :) + hi_lo_ds([1:L]+h/2, :);
hi_hi_ds(1:L, :) = hi_hi_ds(1:L, :) + hi_hi_ds([1:L]+h/2, :);

% crop
cA = lo_lo_ds(1:h/2, :);
cH = lo_hi_ds(1:h/2, :);
cV = hi_lo_ds(1:h/2, :);
cD = hi_hi_ds(1:h/2, :);

% output
cA = cA';
cH = cH';
cV = cV';
cD = cD';

end
