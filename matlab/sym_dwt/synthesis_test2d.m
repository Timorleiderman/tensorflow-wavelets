clear

[af, sf] = farras;
J = 1;                      
L = 3*2^(J+1);
N = L/2^J;
% x = ones(64,64).*(1:64);
x = rand(128,64);
w = mydwt2d(x,J,af(:,1),af(:,2));

rec = myidwt2d(w, J, sf(:,1),sf(:,2));
rec_ref = idwt2D(w, J, sf);

err = max(max(abs(rec-rec_ref)));