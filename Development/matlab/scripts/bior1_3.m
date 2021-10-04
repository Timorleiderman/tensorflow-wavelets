clear

x = 1:1:100;
wavelet_name = 'db4';

[cA,cD] = dwt(x,wavelet_name);
X = idwt(cA,cD, wavelet_name);

[LoD,HiD] = wfilters(wavelet_name,'d');

ca_approx = downsample(filter(LoD,1,x),2,1);
cd_approx = downsample(filter(HiD,1,x),2,1);

x_rec_a = filter(LoD,1,upsample(ca_approx,2));
x_rec_b = filter(HiD,1,upsample(cd_approx,2));

% filter implementation
y_ca = conv(x,LoD,'full');
ca = y_ca(2:2:end-1); % downsample remove edge values

y_cd = conv(x,HiD,'full');
cd = y_cd(2:2:end-1); % downsample remove edge values

y_rec_a = upsample(ca,2);
y_rec_a = conv(y_rec_a,LoD,'full');

y_rec_d = upsample(cd,2);
y_rec_d = conv(y_rec_d,HiD,'full');

y_rec = y_rec_a - y_rec_d;

err_ca = max(cA - ca);
err_cd = max(cD - cd);

X_rec = x_rec_a - x_rec_b;

err_dwt_iwt = max(abs(X-x));
err_my = max(abs(X_rec-x));

figure(1)
subplot(1,2,1)
plot(LoD)
title('lpf')

subplot(1,2,2)
plot(HiD)
title('hpf')

