function w = mydualtree(x, J, FS_LoD_t1, FS_HiD_t1,FS_LoD_t2, FS_HiD_t2, LoD_t1, HiD_t1, LoD_t2, HiD_t2)

% Dual-tree Complex Discrete Wavelet Transform

% normalization
x = x/sqrt(2);

% Tree 1
[x1 w{1}{1}] = analysis_filter_bank(x, FS_LoD_t1, FS_HiD_t1);
for j = 2:J
    [x1 w{j}{1}] = analysis_filter_bank(x1, LoD_t1, HiD_t1);
end
w{J+1}{1} = x1;

% Tree 2
[x2 w{1}{2}] = analysis_filter_bank(x, FS_LoD_t2, FS_HiD_t2);
for j = 2:J
    [x2 w{j}{2}] = analysis_filter_bank(x2, LoD_t2, HiD_t2);
end
w{J+1}{2} = x2;


