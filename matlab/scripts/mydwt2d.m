% 2d wavelet decomposition for db2
% input grayscale image
function [LL, LH, HL, HH] = mydwt2d(img,LoD,HiD)

[w, h] = size(img);
dwt_pad_type = 'sym';
conv_type= 'valid';

% symetric border padding colums for convolution
img_col_pad = wextend('addcol',dwt_pad_type,img,3);

% convlove with low pass and hight pass filters
approx_conv_lp = conv2(img_col_pad,LoD(:)',conv_type);
approx_conv_hp = conv2(img_col_pad,HiD(:)',conv_type);

% downsample
approx_lp_ds = approx_conv_lp(:,2:2:w);
approx_hp_ds = approx_conv_hp(:,2:2:w);

% symetric border padding rows for convolution
approx_lp_ds_row_pad = wextend('addrow',dwt_pad_type,approx_lp_ds,3);
approx_hp_ds_row_pad = wextend('addrow',dwt_pad_type,approx_hp_ds,3);

% convlove with low pass and hight pass filters
approx_lp_lp_conv = conv2(approx_lp_ds_row_pad',LoD(:)',conv_type);
approx_lp_hp_conv = conv2(approx_lp_ds_row_pad',HiD(:)',conv_type);
approx_hp_lp_conv = conv2(approx_hp_ds_row_pad',LoD(:)',conv_type);
approx_hp_hp_conv = conv2(approx_hp_ds_row_pad',HiD(:)',conv_type);

% reorder output shape
approx_lp_lp_conv = approx_lp_lp_conv';
LL = approx_lp_lp_conv(2:2:h,:);

approx_lp_hp_conv = approx_lp_hp_conv';
LH = approx_lp_hp_conv(2:2:h,:);

approx_hp_lp_conv = approx_hp_lp_conv';
HL = approx_hp_lp_conv(2:2:h,:);

approx_hp_hp_conv = approx_hp_hp_conv';
HH = approx_hp_hp_conv(2:2:h,:);
end
    