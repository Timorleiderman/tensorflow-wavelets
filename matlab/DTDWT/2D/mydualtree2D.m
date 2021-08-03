% 2D Dual-Tree Discrete Wavelet Transform
function w = mydualtree2D(x, J, FS_LoD_t1, FS_HiD_t1,FS_LoD_t2, FS_HiD_t2, LoD_t1, HiD_t1, LoD_t2, HiD_t2)

% normalization
x = x/sqrt(2);

% Tree 1
[x1 w{1}{1}{1} w{1}{1}{2} w{1}{1}{3}] = analysis_filter_bank2d(x, FS_LoD_t1, FS_HiD_t1);
for j = 2:J
    [x1 w{j}{1}{1} w{j}{1}{2} w{j}{1}{3}] = analysis_filter_bank2d(x1, LoD_t1, HiD_t1);
end
w{J+1}{1} = x1; 

% Tree 2
[x2 w{1}{2}{1} w{1}{2}{2} w{1}{2}{3}] = analysis_filter_bank2d(x, FS_LoD_t2, FS_HiD_t2);
for j = 2:J
    [x2 w{j}{2}{1} w{j}{2}{2} w{j}{2}{3}] = analysis_filter_bank2d(x2, LoD_t2, HiD_t2);
end
w{J+1}{2} = x2;

% sum and difference
for j = 1:J
    for m = 1:3
        A = w{j}{1}{m};
        B = w{j}{2}{m};
        w{j}{1}{m} = (A+B)/sqrt(2);
        w{j}{2}{m} = (A-B)/sqrt(2);
    end
end

