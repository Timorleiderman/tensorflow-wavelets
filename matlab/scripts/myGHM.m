function [L1L1, L1H1, H1L1, H1H1, L1L2, L1H2, H1L2, H1H2, ...
          L2L1, L2H1, H2L1, H2H1, L2L2, L2H2, H2L2, H2H2] = GHM(img)

dwt_pad_type = 'sym';
conv_type= 'valid';

% initialize the coefficients
H0 = [3/(5*sqrt(2)), 4/5;
     -1/20, -3/(10*sqrt(2))];
 
H1 = [3/(5*sqrt(2)), 0;
      9/20,  1/sqrt(2)];
  
H2 = [0,     0;
      9/20, -3/(10*sqrt(2))];
  
H3 = [0,    0;
     -1/20, 0];
 
G0=[-1/20, -3/(10*sqrt(2));
    1/(10*sqrt(2)), 3/10];

G1=[9/20,   -1/sqrt(2);
    -9/(10*sqrt(2)),0];

G2=[9/20,  -3/(10*sqrt(2));
    9/(10*sqrt(2)), -3/10];

G3=[-1/20,           0;
    -1/(10*sqrt(2)), 0];


% construct the W matrix
H = [H0, H1, H2, H3];
G = [G0, G1, G2, G3];

H_lpf1 = H(1,:);
H_lpf2 = H(2,:);

G_hpf1 = G(1,:);
G_hpf2 = G(2,:);

fl = length(H_lpf1) - 1;
[h, w] = size(img);

img_oversampel(1:2:2*h,:) = img;
img_oversampel(2:2:2*h,:) = img.*1/sqrt(2);

img_oversampel_col_pad = wextend('addcol', dwt_pad_type,img_oversampel,fl);
% img_padd = wextend('addcol', dwt_pad_type, img, fl);

approx_conv_lp1 = conv2(img_oversampel_col_pad, H_lpf1(:), conv_type);
approx_conv_lp2 = conv2(img_oversampel_col_pad, H_lpf2(:), conv_type);
approx_conv_hp1 = conv2(img_oversampel_col_pad, G_hpf1(:), conv_type);
approx_conv_hp2 = conv2(img_oversampel_col_pad, G_hpf2(:), conv_type);
% load("z.mat")
% zz = z(1:4:end,:);
[h_padd, w_padd] = size(approx_conv_lp1);

approx_lp1_ds1 = approx_conv_lp1(:, 1:2:w_padd);
approx_lp2_ds2 = approx_conv_lp2(:, 1:2:w_padd);
approx_hp1_ds1 = approx_conv_hp1(:, 1:2:w_padd);
approx_hp2_ds2 = approx_conv_hp2(:, 1:2:w_padd);

[h_padd_ds, w_padd_ds] = size(approx_lp1_ds1');

approx_lp1_ds_oversampel(1:2:2*h_padd_ds,:) = approx_lp1_ds1';
approx_lp1_ds_oversampel(2:2:2*h_padd_ds,:) = 1/sqrt(2)*approx_lp1_ds1';

approx_lp2_ds_oversampel(1:2:2*h_padd_ds,:) = approx_lp2_ds2';
approx_lp2_ds_oversampel(2:2:2*h_padd_ds,:) = 1/sqrt(2)*approx_lp2_ds2';

approx_hp1_ds_oversampel(1:2:2*h_padd_ds,:) = approx_hp1_ds1';
approx_hp1_ds_oversampel(2:2:2*h_padd_ds,:) = 1/sqrt(2)*approx_hp1_ds1';

approx_hp2_ds_oversampel(1:2:2*h_padd_ds,:) = approx_hp2_ds2';
approx_hp2_ds_oversampel(2:2:2*h_padd_ds,:) = 1/sqrt(2)*approx_hp2_ds2';

% add rows but its already transposed so add col
approx_lp_ds1_row_pad = wextend('addcol', dwt_pad_type, approx_lp1_ds_oversampel,fl);
approx_lp_ds2_row_pad = wextend('addcol', dwt_pad_type, approx_lp2_ds_oversampel,fl);
approx_hp_ds1_row_pad = wextend('addcol', dwt_pad_type, approx_hp1_ds_oversampel,fl);
approx_hp_ds2_row_pad = wextend('addcol', dwt_pad_type, approx_hp2_ds_oversampel,fl);

% convlove with low pass and hight pass filters
approx_lp1_lp1_conv1 = conv2(approx_lp_ds1_row_pad, H_lpf1(:)', conv_type);
approx_lp1_hp1_conv1 = conv2(approx_lp_ds1_row_pad, G_hpf1(:)', conv_type);
approx_hp1_lp1_conv1 = conv2(approx_hp_ds1_row_pad, H_lpf1(:)', conv_type);
approx_hp1_hp1_conv1 = conv2(approx_hp_ds1_row_pad, G_hpf1(:)', conv_type);

approx_lp2_lp2_conv1 = conv2(approx_lp_ds2_row_pad, H_lpf2(:)', conv_type);
approx_lp2_hp2_conv1 = conv2(approx_lp_ds2_row_pad, G_hpf2(:)', conv_type);
approx_hp2_lp2_conv1 = conv2(approx_hp_ds2_row_pad, H_lpf2(:)', conv_type);
approx_hp2_hp2_conv1 = conv2(approx_hp_ds2_row_pad, G_hpf2(:)', conv_type);

[h_padd_os, w_padd_os] = size(approx_lp1_lp1_conv1);

approx_lp1_lp1_conv1 = approx_lp1_lp1_conv1';
LL1 = approx_lp1_lp1_conv1(1:2:w_padd_os,:);

L1L1 = LL1(1:2:end,fl:2:end-fl-1);
L1L2 = LL1(2:2:end,fl+1:2:end-fl);

approx_lp1_hp1_conv1 = approx_lp1_hp1_conv1';
LH1 = approx_lp1_hp1_conv1(1:2:w_padd_os,:);

L1H1 = LH1(1:2:end,fl:2:end-fl-1);
L1H2 = LH1(2:2:end,fl+1:2:end-fl);

approx_hp1_lp1_conv1 = approx_hp1_lp1_conv1';
HL1 = approx_hp1_lp1_conv1(1:2:w_padd_os,:);

H1L1 = HL1(1:2:end,fl:2:end-fl-1);
H1L2 = HL1(2:2:end,fl+1:2:end-fl);

approx_hp1_hp1_conv1 = approx_hp1_hp1_conv1';
HH1 = approx_hp1_hp1_conv1(1:2:w_padd_os,:);

H1H1 = HH1(1:2:end,fl:2:end-fl-1);
H1H2 = HH1(2:2:end,fl+1:2:end-fl);

approx_lp2_lp2_conv1 = approx_lp2_lp2_conv1';
LL2 = approx_lp2_lp2_conv1(1:2:w_padd_os,:);

L2L1 = LL2(1:2:end,fl:2:end-fl-1);
L2L2 = LL2(2:2:end,fl+1:2:end-fl);

approx_lp2_hp2_conv1 = approx_lp2_hp2_conv1';
LH2 = approx_lp2_hp2_conv1(1:2:w_padd_os,:);

L2H1 = LH2(1:2:end,fl:2:end-fl-1);
L2H2 = LH2(2:2:end,fl+1:2:end-fl);

approx_hp2_lp2_conv1 = approx_hp2_lp2_conv1';
HL2 = approx_hp2_lp2_conv1(1:2:w_padd_os,:);

H2L1 = HL2(1:2:end,fl:2:end-fl-1);
H2L2 = HL2(2:2:end,fl+1:2:end-fl);

approx_hp2_hp2_conv1 = approx_hp2_hp2_conv1';
HH2 = approx_hp2_hp2_conv1(1:2:w_padd_os,:);

H2H1 = HH2(1:2:end,fl:2:end-fl-1);
H2H2 = HH2(2:2:end,fl+1:2:end-fl);


end