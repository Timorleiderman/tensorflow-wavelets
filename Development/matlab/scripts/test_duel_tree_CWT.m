close all
clear

x = rand(256);
J = 3;

[Faf, Fsf] = FSfarras;
[af, sf] = dualfilt1;

w = mydueltree2d(x, J, Faf, af);
y = myidualtree2d(w, J, Fsf, sf);
jm
err = x - y;
max(max(abs(err)))