% 2d wavelet decomposition for db2
% input grayscale image
function [img] = myidwt2d(LL,LH,HL,HH,LoR,HiR)

[w, h] = size(LL);
dwt_pad_type = 'sym';
conv_type= 'valid';

fl = length(LoR) - 1;

% symetric border padding colums for convolution
LL_pad = wextend('addcol',dwt_pad_type,LL,fl);
LH_pad = wextend('addcol',dwt_pad_type,LH,fl);
HL_pad = wextend('addcol',dwt_pad_type,HL,fl);
HH_pad = wextend('addcol',dwt_pad_type,HH,fl);

LL_pad = wextend('addrow',dwt_pad_type,LL_pad,fl);
LH_pad = wextend('addrow',dwt_pad_type,LH_pad,fl);
HL_pad = wextend('addrow',dwt_pad_type,HL_pad,fl);
HH_pad = wextend('addrow',dwt_pad_type,HH_pad,fl);

[h_padd, w_padd] = size(LL_pad);

LL_pad_us = zeros(h_padd*2,w_padd*2);
LH_pad_us = zeros(h_padd*2,w_padd*2);
HL_pad_us = zeros(h_padd*2,w_padd*2);
HH_pad_us = zeros(h_padd*2,w_padd*2);

LL_pad_us(1:2:end,1:2:end) = LL_pad;
LH_pad_us(1:2:end,1:2:end) = LH_pad;
HL_pad_us(1:2:end,1:2:end) = HL_pad;
HH_pad_us(1:2:end,1:2:end) = HH_pad;

LL_conv_lpf = conv2(LL_pad_us,LoR(:)',conv_type);
LL_conv_lpf_lpf = conv2(LL_conv_lpf',LoR(:)',conv_type);

LH_conv_lpf = conv2(LH_pad_us,LoR(:)',conv_type);
LH_conv_lpf_hpf = conv2(LH_conv_lpf',HiR(:)',conv_type);

HL_conv_lpf = conv2(HL_pad_us,HiR(:)',conv_type);
HL_conv_hpf_lpf = conv2(HL_conv_lpf',LoR(:)',conv_type);

HH_conv_lpf = conv2(HH_pad_us,HiR(:)',conv_type);
HH_conv_hpf_hpf = conv2(HH_conv_lpf',HiR(:)',conv_type);


LL_recon = LL_conv_lpf_lpf';
LL_recon_o = LL_recon;

LH_recon = LH_conv_lpf_hpf';
LH_recon_o = LH_recon;

HL_recon = HL_conv_hpf_lpf';
HL_recon_o = HL_recon;

HH_recon = HH_conv_hpf_hpf';
HH_recon_o = HH_recon;

img = LL_recon_o + LH_recon_o + HL_recon_o + HH_recon_o;

crop = (fl )*2;
img = img(crop:end-crop,crop:end-crop);

end
    