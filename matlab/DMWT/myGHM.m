function [b] = GHM(img)

dwt_pad_type = 'sym';
conv_type= 'same';

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
w= [H0, H1, H2, H3;
    G0, G1, G2, G3];


H = [H0, H1, H2, H3];
G = [G0, G1, G2, G3];

H_lpf1 = H(1,:);
H_lpf2 = H(2,:);

G_hpf1 = G(1,:);
G_hpf2 = G(2,:);

fl = length(H_lpf1);
[h, w] = size(img);

img_oversampel(1:2:2*h,:) = img;
img_oversampel(2:2:2*h,:) = img.*1/sqrt(2);

L = length(H_lpf1)/2;
% % circular shift 2d
% n = 0:h*2-1;
% n = mod(n+L, h*2);
% img_oversampel = img_oversampel(n+1,:);

% zero pad borders
img_oversampel_row_pad = wextend('addrow','zpd',img_oversampel, fl);

approx_conv_lp1 = conv2(img_oversampel_row_pad, H_lpf1(:), conv_type);
approx_conv_lp2 = conv2(img_oversampel_row_pad, H_lpf2(:), conv_type);
approx_conv_hp1 = conv2(img_oversampel_row_pad, G_hpf1(:), conv_type);
approx_conv_hp2 = conv2(img_oversampel_row_pad, G_hpf2(:), conv_type);

load("z.mat")
zz1 = z(1:4:end,:);
zz2 = z(2:4:end,:);
zz3 = z(3:4:end,:);
zz4 = z(4:4:end,:);

[h_padd, w_padd] = size(approx_conv_lp1);


load("L1L1_ref.mat")
load("L1L2_ref.mat")
load("L2L1_ref.mat")
load("L2L2_ref.mat")


% down sample
approx_lp1_ds = approx_conv_lp1(fl-1:2:h_padd-2,: );
approx_lp1_ds1 = approx_lp1_ds(1:2:end-L-1,: );

approx_lp2_ds = approx_conv_lp2(fl-1:2:h_padd-2,: );
approx_lp2_ds1 = approx_lp2_ds(L-1:2:end-L+1,: );

approx_hp1_ds = approx_conv_hp1(fl-1:2:h_padd-2,: );
approx_hp1_ds1 = approx_hp1_ds(L-1:2:end-L+1,: );

approx_hp2_ds = approx_conv_hp2(fl-1:2:h_padd-2,: );
approx_hp2_ds1 = approx_hp2_ds(L-1:2:end-L+1,: ).*(-1);


err_lp1_zz1 = max(max(approx_lp1_ds1 - zz1 ));
err_lp2_zz2 = max(max(approx_lp2_ds1 - zz2 ));
err_hp1_zz3 = max(max(approx_hp1_ds1 - zz3 ));
err_hp2_zz4 = max(max(approx_hp2_ds1 - zz4 ));


[h_padd_ds, w_padd_ds] = size(approx_lp1_ds1');

approx_lp1_ds_oversampel(1:2:2*h_padd_ds,:) = approx_lp1_ds1';
approx_lp1_ds_oversampel(2:2:2*h_padd_ds,:) = 1/sqrt(2)*approx_lp1_ds1';

approx_lp2_ds_oversampel(1:2:2*h_padd_ds,:) = approx_lp2_ds1';
approx_lp2_ds_oversampel(2:2:2*h_padd_ds,:) = 1/sqrt(2)*approx_lp2_ds1';

approx_hp1_ds_oversampel(1:2:2*h_padd_ds,:) = approx_hp1_ds1';
approx_hp1_ds_oversampel(2:2:2*h_padd_ds,:) = 1/sqrt(2)*approx_hp1_ds1';

approx_hp2_ds_oversampel(1:2:2*h_padd_ds,:) = approx_hp2_ds1';
approx_hp2_ds_oversampel(2:2:2*h_padd_ds,:) = 1/sqrt(2)*approx_hp2_ds1';

% add rows but its already transposed so add col
approx_lp1_ds_row_pad = wextend('addrow', dwt_pad_type, approx_lp1_ds_oversampel,fl);
approx_lp2_ds_row_pad = wextend('addrow', dwt_pad_type, approx_lp2_ds_oversampel,fl);
approx_hp1_ds_row_pad = wextend('addrow', dwt_pad_type, approx_hp1_ds_oversampel,fl);
approx_hp2_ds_row_pad = wextend('addrow', dwt_pad_type, approx_hp2_ds_oversampel,fl);

% convlove with low pass and hight pass filters
approx_lp1_lp1_conv1 = conv2(approx_lp1_ds_row_pad, H_lpf1(:), conv_type);
approx_lp1_hp1_conv1 = conv2(approx_lp1_ds_row_pad, G_hpf1(:), conv_type);
approx_hp1_lp1_conv1 = conv2(approx_hp1_ds_row_pad, H_lpf1(:), conv_type);
approx_hp1_hp1_conv1 = conv2(approx_hp1_ds_row_pad, G_hpf1(:), conv_type);

approx_lp1_lp2_conv1 = conv2(approx_lp1_ds_row_pad, H_lpf2(:), conv_type);
approx_lp1_hp2_conv1 = conv2(approx_lp1_ds_row_pad, G_hpf2(:), conv_type);
approx_hp1_lp2_conv1 = conv2(approx_hp1_ds_row_pad, H_lpf2(:), conv_type);
approx_hp1_hp2_conv1 = conv2(approx_hp1_ds_row_pad, G_hpf2(:), conv_type);

approx_lp2_lp1_conv1 = conv2(approx_lp2_ds_row_pad, H_lpf1(:), conv_type);
approx_lp2_hp1_conv1 = conv2(approx_lp2_ds_row_pad, G_hpf1(:), conv_type);
approx_hp2_lp1_conv1 = conv2(approx_hp2_ds_row_pad, H_lpf1(:), conv_type);
approx_hp2_hp1_conv1 = conv2(approx_hp2_ds_row_pad, G_hpf1(:), conv_type);

approx_lp2_lp2_conv1 = conv2(approx_lp2_ds_row_pad, H_lpf2(:), conv_type);
approx_lp2_hp2_conv1 = conv2(approx_lp2_ds_row_pad, G_hpf2(:), conv_type);
approx_hp2_lp2_conv1 = conv2(approx_hp2_ds_row_pad, H_lpf2(:), conv_type);
approx_hp2_hp2_conv1 = conv2(approx_hp2_ds_row_pad, G_hpf2(:), conv_type);


% 1 1
approx_lp1_lp1_conv1_ds = approx_lp1_lp1_conv1(L-1:2:end,:);
L1L1 = approx_lp1_lp1_conv1_ds(3:2:end-L-1,:)';

err_L1L1= max(max(L1L1_ref - L1L1));

approx_lp1_hp1_conv1_ds = approx_lp1_hp1_conv1(L-1:2:end,:);
L1H1 = approx_lp1_hp1_conv1_ds(3:2:end-L-1,:)';
approx_hp1_lp1_conv1_ds = approx_hp1_lp1_conv1(L-1:2:end,:);
H1L1 = approx_hp1_lp1_conv1_ds(3:2:end-L-1,:)';
approx_hp1_hp1_conv1_ds = approx_hp1_hp1_conv1(L-1:2:end,:);
H1H1 = approx_hp1_hp1_conv1_ds(3:2:end-L-1,:)';

% 1 2
approx_lp1_lp2_conv1_ds = approx_lp1_lp2_conv1(L-1:2:end,:);
L1L2 = approx_lp1_lp2_conv1_ds(L+1:2:end-L,:)';
err_L1L2= max(max(L1L2_ref - L1L2));
approx_lp1_hp2_conv1_ds = approx_lp1_hp2_conv1(L-1:2:end,:);
L1H2 = approx_lp1_hp2_conv1_ds(3:2:end-L-1,:)';
approx_hp1_lp2_conv1_ds = approx_hp1_lp2_conv1(L-1:2:end,:);
H1L2 = approx_hp1_lp2_conv1_ds(3:2:end-L-1,:)';
approx_hp1_hp2_conv1_ds = approx_hp1_hp2_conv1(L-1:2:end,:);
H1H2 = approx_hp1_hp2_conv1_ds(3:2:end-L-1,:)';




% 2 1
approx_lp2_lp1_conv1_ds = approx_lp2_lp1_conv1(L-1:2:end,:);
L2L1 = approx_lp2_lp1_conv1_ds(3:2:end-L-1,:)';
approx_lp2_hp1_conv1_ds = approx_lp2_hp1_conv1(L-1:2:end,:);
L2H1 = approx_lp2_hp1_conv1_ds(3:2:end-L-1,:)';
approx_hp2_lp1_conv1_ds = approx_hp2_lp1_conv1(L-1:2:end,:);
H2L1 = approx_hp2_lp1_conv1_ds(3:2:end-L-1,:)';
approx_hp2_hp1_conv1_ds = approx_hp2_hp1_conv1(L-1:2:end,:);
H2H1 = approx_hp2_hp1_conv1_ds(3:2:end-L-1,:)';


approx_lp2_lp2_conv1_ds = approx_lp2_lp2_conv1(L-1:2:end,:);
L2L2 = approx_lp2_lp2_conv1_ds(3:2:end-L-1,:)';
approx_lp2_hp2_conv1_ds = approx_lp2_hp2_conv1(L-1:2:end,:);
L2H2 = approx_lp2_hp2_conv1_ds(3:2:end-L-1,:)';
approx_hp2_lp2_conv1_ds = approx_hp2_lp2_conv1(L-1:2:end,:);
H2L2 = approx_hp2_lp2_conv1_ds(3:2:end-L-1,:)';
approx_hp2_hp2_conv1_ds = approx_hp2_hp2_conv1(L-1:2:end,:);
H2H2 = approx_hp2_hp2_conv1_ds(3:2:end-L-1,:)';



err_L2L1= max(max(L2L1_ref - L2L1));
err_L2L2= max(max(L2L2_ref - L2L2));

LL = [
     [L1L1,L2L1  ] ;
     [L1L2,L2L2]];
 
LH = [
     [L1H1,L2H1  ] ;
     [L1H2, L2H2]];
HL = [
     [H1L1,H2L1  ] ;
     [H1L2, H2L2]];
HH = [
     [H1H1,H1H2 ] ;
     [H2H1, H2H2]];

b = [
    [LL, LH ] ;
    [HL, HH]];

end