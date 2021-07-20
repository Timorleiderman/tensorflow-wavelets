clear

addpath utils

f = imread('../../input/LennaGrey.png');
f = double(f);

b = GHM(f);
recon_img = IGHM(b);

disperr2d(f,recon_img);

MWS("cl")