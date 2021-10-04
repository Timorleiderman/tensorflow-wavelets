% 2d wavelet decomposition for db2
% input grayscale image
function [LL, LH, HL, HH] = mydwt2d(img,LoD,HiD)

[h, w] = size(img);
dwt_pad_type = 'sym';
conv_type= 'valid';

fl = length(LoD) - 1;

out_w = (w/2) + (length(LoD)/2) -1;
out_h = (h/2) + (length(LoD)/2) -1;

% symetric border padding colums for convolution
img_col_pad = wextend('addcol', dwt_pad_type,img,fl);

% convlove with low pass and hight pass filters
approx_conv_lp = conv2(img_col_pad, LoD(:)', conv_type);
approx_conv_hp = conv2(img_col_pad, HiD(:)', conv_type);

[h_padd, w_padd] = size(approx_conv_lp);

% downsample
approx_lp_ds = approx_conv_lp(:, 2:2:w_padd);
approx_hp_ds = approx_conv_hp(:, 2:2:w_padd);

% symetric border padding rows for convolution
approx_lp_ds_row_pad = wextend('addrow', dwt_pad_type, approx_lp_ds,fl);
approx_hp_ds_row_pad = wextend('addrow', dwt_pad_type, approx_hp_ds,fl);


% convlove with low pass and hight pass filters
approx_lp_lp_conv = conv2(approx_lp_ds_row_pad', LoD(:)', conv_type);
approx_lp_hp_conv = conv2(approx_lp_ds_row_pad', HiD(:)', conv_type);
approx_hp_lp_conv = conv2(approx_hp_ds_row_pad', LoD(:)', conv_type);
approx_hp_hp_conv = conv2(approx_hp_ds_row_pad', HiD(:)', conv_type);



% reorder output shape
approx_lp_lp_conv = approx_lp_lp_conv';
[h_padd, w_padd] = size(approx_lp_lp_conv);

LL = approx_lp_lp_conv(2:2:h_padd,:);

approx_lp_hp_conv = approx_lp_hp_conv';
LH = approx_lp_hp_conv(2:2:h_padd,:);

approx_hp_lp_conv = approx_hp_lp_conv';
HL = approx_hp_lp_conv(2:2:h_padd,:);

approx_hp_hp_conv = approx_hp_hp_conv';
HH = approx_hp_hp_conv(2:2:h_padd,:);
end
    