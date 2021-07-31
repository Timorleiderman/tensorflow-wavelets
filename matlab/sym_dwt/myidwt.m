
function y = myidwt(w, J, LoR, HiR)

y = w{J+1};
for k = J:-1:1
   y = synthesis_filter_bank(y, w{k}, LoR, HiR);
end

end