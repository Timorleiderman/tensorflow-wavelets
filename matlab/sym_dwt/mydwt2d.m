function w = mydwt2d(x, J, LoD, HiD)

for k = 1:J
    
    [x, H] = analysis_filter_bank2d(x, LoD(:), HiD(:));

    [x, w{k}{1}] = analysis_filter_bank2d(x', LoD(:), HiD(:));
    [w{k}{2}, w{k}{3}] = analysis_filter_bank2d(H', LoD(:), HiD(:));

x = x';
w{k}{1} = w{k}{1}';
w{k}{2} = w{k}{2}';
w{k}{3} = w{k}{3}';

end
w{J+1} = x;

end