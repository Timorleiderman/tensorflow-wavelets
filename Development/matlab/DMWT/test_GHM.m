clear
close all

addpath ../utils

f = imread('../../input/LennaGrey.png');
f = double(f);
[N,M]=size(f);

b = GHM(f);
recon_img = IGHM(b);

b1=b(1:N,1:N);
b2=b(1:N,N+1:2*N);
b3=b(N+1:2*N,1:N);
b4=b(N+1:2*N,N+1:2*N);

L1L1_ref = b1(1:256,1:256);
L2L1_ref = b1(257:512,1:256);
L1L2_ref = b1(1:256,257:512);
L2L2_ref = b1(257:512,257:512);

L1H1_ref = b2(1:256,1:256);
L2H1_ref = b2(257:512,1:256);
L1H2_ref = b2(1:256,257:512);
L2H2_ref = b2(257:512,257:512);

H1L1_ref = b3(1:256,1:256);
H2L1_ref = b3(257:512,1:256);
H1L2_ref = b3(1:256,257:512);
H2L2_ref = b3(257:512,257:512);


H1H1_ref = b4(1:256,1:256);
H2H1_ref = b4(257:512,1:256);
H1H2_ref = b4(1:256,257:512);
H2H2_ref = b4(257:512,257:512);

[b_my] = myGHM(f);
b_my1=b_my(1:N,1:N);
b_my2=b_my(1:N,N+1:2*N);
b_my3=b_my(N+1:2*N,1:N);
b_my4=b_my(N+1:2*N,N+1:2*N);
disperr2d(b_my1,b1);
disperr2d(b_my2,b2);
disperr2d(b_my3,b3);
disperr2d(b_my4,b4);

figure(2)
imshow(uint8(b))

figure(1)
imshow(uint8(b_my));

disperr2d(b_my,b);

%MWS("ghm");