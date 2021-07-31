close all
clear

J = 2;
L = 3*2^(J+1);
N = L/2^J;
Y = zeros(2*L,6*L);
Y = zeros(L,L);
[A,D] = dualtree2(Y,'Level',J);