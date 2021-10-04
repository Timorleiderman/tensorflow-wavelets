function [b] = GHM(img)

dwt_pad_type = 'zpd';
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
w_mat= [H0, H1, H2, H3;
        G0, G1, G2, G3];


H = [H0, H1, H2, H3];
G = [G0, G1, G2, G3];

H_lpf1 = H(1,:);
H_lpf2 = H(2,:);

G_hpf1 = G(1,:);
G_hpf2 = G(2,:);

fl = length(H_lpf1);
[h, w] = size(img);

img_os(1:2:2*h,:) = img;
img_os(2:2:2*h,:) = img.*1/sqrt(2);

L = length(H_lpf1)/2;

% zero pad borders
img_os_row_pad = wextend('addrow',dwt_pad_type,img_os, fl);

conv_lp1 = conv2(img_os_row_pad, H_lpf1(:), conv_type);
conv_lp2 = conv2(img_os_row_pad, H_lpf2(:), conv_type);
conv_hp1 = conv2(img_os_row_pad, G_hpf1(:), conv_type);
conv_hp2 = conv2(img_os_row_pad, G_hpf2(:), conv_type);

load("z.mat")
zz1 = z(1:4:end,:);
zz2 = z(2:4:end,:);
zz3 = z(3:4:end,:);
zz4 = z(4:4:end,:);

[h_padd, w_padd] = size(conv_lp1);


load("L1L1_ref.mat")
load("L1L2_ref.mat")
load("L2L1_ref.mat")
load("L2L2_ref.mat")

load("L1H1_ref.mat")
load("L1H2_ref.mat")
load("L2H1_ref.mat")
load("L2H2_ref.mat")

load("H1L1_ref.mat")
load("H1L2_ref.mat")
load("H2L1_ref.mat")
load("H2L2_ref.mat")

load("H1H1_ref.mat")
load("H1H2_ref.mat")
load("H2H1_ref.mat")
load("H2H2_ref.mat")


% ii=0:4:2*N-1;
% jj=sort([ii+1,ii+2]);
% kk=sort([ii+3,ii+4]);
% p=[p;z(jj,:);z(kk,:)];

% down sample
lp1_ds = conv_lp1(fl-1:2:h_padd-2,: );
lp1_ds1 = lp1_ds(1:2:end-L-1,: );

lp2_ds = conv_lp2(fl-1:2:h_padd-2,: );
lp2_ds1 = lp2_ds(L-1:2:end-L+1,: );

hp1_ds = conv_hp1(fl-1:2:h_padd-2,: );
hp1_ds1 = hp1_ds(L-1:2:end-L+1,: );

hp2_ds = conv_hp2(fl-1:2:h_padd-2,: );
hp2_ds1 = hp2_ds(L-1:2:end-L+1,: ).*(-1);

err_lp1_zz1 = max(max(lp1_ds1 - zz1 ));
err_lp2_zz2 = max(max(lp2_ds1 - zz2 ));
err_hp1_zz3 = max(max(hp1_ds1 - zz3 ));
err_hp2_zz4 = max(max(hp2_ds1 - zz4 ));

[h_padd_ds, w_padd_ds] = size(lp1_ds1');


p=[lp1_ds;lp2_ds];



lp1_ds_os(1:2:2*h_padd_ds,:) = lp1_ds1';
lp1_ds_os(2:2:2*h_padd_ds,:) = lp1_ds1'.*1/sqrt(2);

lp2_ds_os(1:2:2*h_padd_ds,:) = lp2_ds1';
lp2_ds_os(2:2:2*h_padd_ds,:) = lp2_ds1'.*1/sqrt(2);

hp1_ds_os(1:2:2*h_padd_ds,:) = hp1_ds1';
hp1_ds_os(2:2:2*h_padd_ds,:) = hp1_ds1'.*1/sqrt(2);

hp2_ds_os(1:2:2*h_padd_ds,:) = hp2_ds1';
hp2_ds_os(2:2:2*h_padd_ds,:) = hp2_ds1'.*1/sqrt(2);

% add rows but its already transposed so add col

lp1_ds_os_row_pad = wextend('addrow', dwt_pad_type, lp1_ds_os,L);
lp2_ds_os_row_pad = wextend('addrow', dwt_pad_type, lp2_ds_os,L);
hp1_ds_os_row_pad = wextend('addrow', dwt_pad_type, hp1_ds_os,L);
hp2_ds_os_row_pad = wextend('addrow', dwt_pad_type, hp2_ds_os,L);

% convlove with low pass and hight pass filters
lp1_lp1_conv = conv2(lp1_ds_os_row_pad, H_lpf1(:), conv_type);
lp1_hp1_conv = conv2(lp1_ds_os_row_pad, G_hpf1(:), conv_type);
hp1_lp1_conv = conv2(hp1_ds_os_row_pad, H_lpf1(:), conv_type);
hp1_hp1_conv = conv2(hp1_ds_os_row_pad, G_hpf1(:), conv_type);

lp1_lp2_conv = conv2(lp1_ds_os_row_pad, H_lpf2(:), conv_type);
lp1_hp2_conv = conv2(lp1_ds_os_row_pad, G_hpf2(:), conv_type);
hp1_lp2_conv = conv2(hp1_ds_os_row_pad, H_lpf2(:), conv_type);
hp1_hp2_conv = conv2(hp1_ds_os_row_pad, G_hpf2(:), conv_type);

lp2_lp1_conv = conv2(lp2_ds_os_row_pad, H_lpf1(:), conv_type);
lp2_hp1_conv = conv2(lp2_ds_os_row_pad, G_hpf1(:), conv_type);
hp2_lp1_conv = conv2(hp2_ds_os_row_pad, H_lpf1(:), conv_type);
hp2_hp1_conv = conv2(hp2_ds_os_row_pad, G_hpf1(:), conv_type);

lp2_lp2_conv = conv2(lp2_ds_os_row_pad, H_lpf2(:), conv_type);
lp2_hp2_conv = conv2(lp2_ds_os_row_pad, G_hpf2(:), conv_type);
hp2_lp2_conv = conv2(hp2_ds_os_row_pad, H_lpf2(:), conv_type);
hp2_hp2_conv = conv2(hp2_ds_os_row_pad, G_hpf2(:), conv_type);

% 1 1

L1L1 = lp1_lp1_conv(L-1:4:end-L-2,:)';
L1H1 = lp1_hp1_conv(L+3:4:end-L-1,:)';
H1L1 = hp1_lp1_conv(L-1:4:end-L-2,:)';
H1H1 = hp1_hp1_conv(L+3:4:end-L-1,:)';

% 1 2

L1L2 = lp1_lp2_conv(L+3:4:end-L-1,:)';
L1H2 = lp1_hp2_conv(L+3:4:end-L-1,:)'.*(-1);
H1L2 = hp1_lp2_conv(L+3:4:end-L-1,:)';
H1H2 = hp1_hp2_conv(L+3:4:end-L-1,:)'.*(-1);

% 2 1
L2L1 = lp2_lp1_conv(L-1:4:end-L-2,:)';
L2H1 = lp2_hp1_conv(L+3:4:end-L-1,:)';
H2L1 = hp2_lp1_conv(L-1:4:end-L-2,:)'.*(-1);
H2H1 = hp2_hp1_conv(L+3:4:end-L-1,:)'*(-1);

L2L2 = lp2_lp2_conv(L+3:4:end-L-1,:)';
L2H2 = lp2_hp2_conv(L+3:4:end-L-1,:)'.*(-1);
H2L2 = hp2_lp2_conv(L+3:4:end-L-1,:)'.*(-1);
H2H2 = hp2_hp2_conv(L+3:4:end-L-1,:)';

err_L1L1 = max(max(L1L1_ref - L1L1));
err_L1L2 = max(max(L1L2_ref - L1L2));
err_L2L1 = max(max(L2L1_ref - L2L1));
err_L2L2 = max(max(L2L2_ref - L2L2));

err_L1H1 = max(max(L1H1_ref - L1H1));
err_L1H2 = max(max(L1H2_ref - L1H2));
err_L2H1 = max(max(L2H1_ref - L2H1));
err_L2H2 = max(max(L2H2_ref - L2H2));

err_H1L1 = max(max(H1L1_ref - H1L1));
err_H1L2 = max(max(H1L2_ref - H1L2));
err_H2L1 = max(max(H2L1_ref - H2L1));
err_H2L2 = max(max(H2L2_ref - H2L2));

err_H1H1 = max(max(H1H1_ref - H1H1));
err_H1H2 = max(max(H1H2_ref - H1H2));
err_H2H1 = max(max(H2H1_ref - H2H1));
err_H2H2 = max(max(H2H2_ref - H2H2));

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