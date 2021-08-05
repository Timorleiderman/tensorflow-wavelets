
close all
clear
addpath DTCWT/2D 
addpath utils
addpath scripts

x = single(imread('../input/LennaGrey.png'));

J = 2;
[Faf, Fsf] = FSfarras;
[af, sf] = dualfilt1;

w = mycplxdual2D(x, J, Faf, af);

 debug_raw(w);
y = myicplxdual2D(w, J, Fsf, sf);


err = max(max(abs(y - x))); 
