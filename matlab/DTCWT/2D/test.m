
clear
x = rand(256);
J = 5;
[Faf, Fsf] = FSfarras;
[af, sf] = dualfilt1;

w = mycplxdual2D(x, J, Faf, af);
y = myicplxdual2D(w, J, Fsf, sf);

err = max(max(abs(y - x))); 