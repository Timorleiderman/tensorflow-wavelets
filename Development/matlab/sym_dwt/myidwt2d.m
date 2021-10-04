function y = myidwt2D(w, J, LoR, HiR)

y = w{J+1};
for k = J:-1:1
   y = synthesis_filter_bank2d(y, w{k}, LoR, HiR);
end

