function y = myidualtree(w, J, FS_LoR_t1, FS_HiR_t1,FS_LoR_t2, FS_HiR_t2, LoR_t1, HiR_t1, LoR_t2, HiR_t2);

% Inverse Dual-tree Complex DWT
% Tree 1
y1 = w{J+1}{1};
for j = J:-1:2
   y1 = synthesis_filter_bank(y1, w{j}{1}, LoR_t1, HiR_t1);
end
y1 = synthesis_filter_bank(y1, w{1}{1}, FS_LoR_t1, FS_HiR_t1);

% Tree 2
y2 = w{J+1}{2};
for j = J:-1:2
   y2 = synthesis_filter_bank(y2, w{j}{2}, LoR_t2, HiR_t2);
end
y2 = synthesis_filter_bank(y2, w{1}{2}, FS_LoR_t2, FS_HiR_t2);

% normalization
y = (y1 + y2)/sqrt(2);

