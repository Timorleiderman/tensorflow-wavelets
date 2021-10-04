close all
clear

[af, sf] = farras;
x = ones(64,64).*(1:64);

[LL, LH, HL, HH] = analysis_filter_bank2d(x, af(:,1), af(:,2));


[L1, H1] = afb2D_A(x, af, 1);
[LL1,    LH1] = afb2D_A(L1, af, 2);
[HL1, HH1] = afb2D_A(H1, af, 2);



err_LL = max(max(abs(LL1-LL)));
err_LH = max(max(abs(LH1-LH)));
err_HL = max(max(abs(HL1-HL)));
err_HH = max(max(abs(HH1-HH)));
