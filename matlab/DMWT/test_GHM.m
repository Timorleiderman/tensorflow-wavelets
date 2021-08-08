clear
close all

addpath utils

f = imread('../../input/LennaGrey.png');
f = double(f);


b = GHM(f);
recon_img = IGHM(b);


[b_my] = myGHM(f);

figure(2)
imshow(uint8(b/2))

figure(1)
imshow(uint8(b_my));

%disperr2d(b_my,b);

%MWS("ghm");