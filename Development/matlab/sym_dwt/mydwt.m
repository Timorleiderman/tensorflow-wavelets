function w = mydwt(x, J, LoD, HiD)

for k = 1:J
    [x w{k}] = analysis_filter_bank(x, LoD, HiD);
end
w{J+1} = x;

end