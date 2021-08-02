
clear
[af, sf] = farras;

LoD = af(:,1);
HiD = af(:,2);

x = ones(64,64).*(1:64);
J = 1;
w = dwt2D(x,J,af);

w_my = mydwt2d(x, J, LoD, HiD);

err_LL = max(max(abs(w{2} - w_my{2})));
err_LH = max(max(abs(w{1}{1} - w_my{1}{1})));
err_HL = max(max(abs(w{1}{2} - w_my{1}{2})));
err_HH = max(max(abs(w{1}{3} - w_my{1}{3})));