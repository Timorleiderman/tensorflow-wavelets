clear
close all

addpath utils

f = imread('../input/LennaGrey.png');
f = double(f);


b = GHM(f);
recon_img = IGHM(b);
figure(2)
imshow(uint8(b/2))

[L1L1, L1H1, H1L1, H1H1, L1L2, L1H2, H1L2, H1H2, ...
    L2L1, L2H1, H2L1, H2H1, L2L2, L2H2, H2L2, H2H2] = myGHM(f);

LL = [L1L1, L1L2;
      L2L1, L2L2];

LH = [L1H1, L1H2;
      L2H1, L2H2];
  
HL = [H1L1, H1L2;
      H2L1, H2L2];
  
HH = [H1H1, H1H2;
      H2H1, H2H2];
  
b_my = [LL,LH;
        HL,HH];
    
figure(1)
imshow(uint8(b_my));

disperr2d(b_my,b);

%MWS("ghm");