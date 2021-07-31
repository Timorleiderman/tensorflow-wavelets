clear
[af, sf] = farras;        % analysis and synthesis filter
x = rand(1,64);           % create generic signal
x = 1:64;           % create generic signal
[lo, hi] = afb(x, af);    % analysis filter bank
[cA, cD] = analysis_filter_bank(x, af(:,1), af(:,2));

y = sfb(lo, hi, sf);      % synthesis filter bank
y_my = synthesis_filter_bank(cA, cD, sf(:,1), sf(:,2));
err = x - y;              % compute error signal
err_my = max(x - y_my);              % compute error signal

err_my_Y = max(y-y_my)
