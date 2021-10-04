clear

x = 1:1:10;

[cA,cD] = dwt(x,'haar');
X = idwt(cA,cD, 'haar');

% filter downsample filter upsample implementation
b_lpf_a = [1/sqrt(2) 1/sqrt(2)];
b_hpf_a = [-1/sqrt(2) 1/sqrt(2)];

b_lpf_s = b_lpf_a;
b_hpf_s = -b_hpf_a;

ca_approx = downsample(filter(b_lpf_a,1,x),2,1);
cd_approx = downsample(filter(b_hpf_a,1,x),2,1);

up_sample_ca_approx = upsample(ca_approx,2);
up_sample_cd_approx = upsample(cd_approx,2);

x_rec_a = filter(b_lpf_s,1,up_sample_ca_approx);
x_rec_b = filter(b_hpf_s,1,up_sample_cd_approx);

X_rec = x_rec_a + x_rec_b;

err_dwt_iwt = max(abs(X-x));
err_filter = max(abs(X_rec-x));
