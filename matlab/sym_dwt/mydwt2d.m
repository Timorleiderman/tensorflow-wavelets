function w = mydwt2d(x, J, LoD, HiD)

for k = 1:J
    
    [x, w{k}{1}, w{k}{2}, w{k}{3}] = analysis_filter_bank2d(x, LoD(:), HiD(:));

end
w{J+1} = x;

end