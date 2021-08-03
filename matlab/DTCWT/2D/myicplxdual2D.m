function y = myicplxdual2D(w, J, Fsf, sf)

% Inverse Dual-Tree Complex 2D Discrete Wavelet Transform

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

y = zeros(size(w{1}{1}{1}{1})*2);
for m = 1:2
    for n = 1:2
        lo = w{J+1}{m}{n};
        for j = J:-1:2
            lo = synthesis_filter_bank2d(lo, w{j}{m}{n}, sf{m}(:,1), sf{m}(:,2), sf{n}(:,1), sf{n}(:,2));
        end
        lo = synthesis_filter_bank2d(lo, w{1}{m}{n}, Fsf{m}(:,1), Fsf{m}(:,2), Fsf{n}(:,1), Fsf{n}(:,2));
        y = y + lo;
    end
end

% normalization
y = y/2;

