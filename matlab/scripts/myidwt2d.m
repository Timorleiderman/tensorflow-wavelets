% 2d wavelet decomposition for db2
% input grayscale image
function [img] = myidwt2d(LL,LH,HL,HH,LoR,HiR)

[w, h] = size(LL);
dwt_pad_type = 'sym';
conv_type= 'valid';

% symetric border padding colums for convolution
LL_pad = wextend('addcol',dwt_pad_type,LL,3);
LH_pad = wextend('addcol',dwt_pad_type,LH,3);
HL_pad = wextend('addcol',dwt_pad_type,HL,3);
HH_pad = wextend('addcol',dwt_pad_type,HH,3);

LL_pad = wextend('addrow',dwt_pad_type,LL_pad,3);
LH_pad = wextend('addrow',dwt_pad_type,LH_pad,3);
HL_pad = wextend('addrow',dwt_pad_type,HL_pad,3);
HH_pad = wextend('addrow',dwt_pad_type,HH_pad,3);

LL_pad_us = zeros(2*w+12,2*h+12);
LH_pad_us = zeros(2*w+12,2*h+12);
HL_pad_us = zeros(2*w+12,2*h+12);
HH_pad_us = zeros(2*w+12,2*h+12);

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
LL_recon_o = LL_recon(5:end-5,5:end-5);

LH_recon = LH_conv_lpf_hpf';
LH_recon_o = LH_recon(5:end-5,5:end-5);

HL_recon = HL_conv_hpf_lpf';
HL_recon_o = HL_recon(5:end-5,5:end-5);

HH_recon = HH_conv_hpf_hpf';
HH_recon_o = HH_recon(5:end-5,5:end-5);


img = LL_recon_o + LH_recon_o + HL_recon_o + HH_recon_o;

end
    