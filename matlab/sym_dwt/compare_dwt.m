close all
clear

[af, sf] = farras;     % analysis and synthesis filter
x = rand(1,64);        % create generic signal
w = dwt(x,3,af);       % analysis filter banks (3 stages)

w_my = mydwt(x,3,af(:,1),af(:,2));

LL = w{1};
LH = w{2};
HL = w{3};
HH = w{4};

LL_my = w_my{1};
LH_my = w_my{2};
HL_my = w_my{3};
HH_my = w_my{4};

err_LL = max(LL-LL_my);
err_LH = max(LH-LH_my);
err_HL = max(HL-HL_my);
err_HH = max(HH-HH_my);