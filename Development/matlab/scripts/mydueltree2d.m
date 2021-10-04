function w = mydueltree2d(x, J, Faf, af)

% 2D Dual-Tree Discrete Wavelet Transform
%
% w = dualtree2D(x, J, Faf, af)
% INPUT:
%    x   - 2-D signal
%    J   - number of stages
%    Faf - first stage filters
%    af  - filters for remaining stages
% OUPUT:
%    w{i}{1:J+1}: tree i wavelet coeffs (i = 1,2)

[x1 w{1}{1}] = afb2D(x, Faf{1});
for k = 2:J
    [x1 w{k}{1}] = afb2D(x1, af{1});
end
w{J+1}{1} = x1;

[x2 w{1}{2}] = afb2D(x, Faf{2});
for k = 2:J
    [x2 w{k}{2}] = afb2D(x2, af{2});
end
w{J+1}{2} = x2;

for k = 1:J
    for m = 1:3
        [w{k}{1}{m} w{k}{2}{m}] = pm(w{k}{1}{m},w{k}{2}{m});
    end
end