function w = mycplxdual2D(x, J, Faf, af)

% Dual-Tree Complex 2D Discrete Wavelet Transform
% normalization
x = x/2;

  
for m = 1:2
    for n = 1:2
        [lo, w{1}{m}{n}{1}, w{1}{m}{n}{2}, w{1}{m}{n}{3}] = analysis_filter_bank2d(x, Faf{m}(:,1), Faf{m}(:,2),Faf{n}(:,1), Faf{n}(:,2));
        for j = 2:J

            [lo, w{j}{m}{n}{1}, w{j}{m}{n}{2}, w{j}{m}{n}{3}] = analysis_filter_bank2d(lo, af{m}(:,1), af{m}(:,2),af{n}(:,1), af{n}(:,2));
            
        end
        w{J+1}{m}{n} = lo;
    end
end

for j = 1:J
    for m = 1:3
            a = w{j}{1}{1}{m};
            b = w{j}{2}{2}{m};
            w{j}{1}{1}{m} = (a + b)/sqrt(2);
            w{j}{2}{2}{m} = (a - b)/sqrt(2);
            
            a = w{j}{1}{2}{m};
            b = w{j}{2}{1}{m};
            w{j}{1}{2}{m} = (a + b)/sqrt(2);
            w{j}{2}{1}{m} = (a - b)/sqrt(2);
    end
end

