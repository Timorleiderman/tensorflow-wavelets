clear

x = 1:1:10;

[cA,cD] = dwt(x,'haar');
X = idwt(cA,cD, 'haar');

% filter downsample filter upsample implementation
b_lpf = [1/sqrt(2) 1/sqrt(2)];
b_hpf = [1/sqrt(2) -1/sqrt(2)];

ca_approx = downsample(filter(b_lpf,1,x),2,1);
cd_approx = downsample(filter(b_hpf,1,x),2,1);

x_rec_a = filter(b_lpf,1,upsample(ca_approx,2));
x_rec_b = filter(b_hpf,1,upsample(cd_approx,2));

X_rec = x_rec_a - x_rec_b;

% filter conv downsampling upsampling conv implementation

y_ca = conv(x,b_lpf,'full');
ca = y_ca(2:2:end-1); % downsample remove edge values
y_cd = conv(x,b_hpf,'full');
cd = y_cd(2:2:end-1); % downsample remove edge values

x_rec_a_reg = zeros(1,length(x)-1);
x_rec_a_reg(1:2:end) = ca;
x_rec_a_reg = conv(x_rec_a_reg,b_lpf,'full');

x_rec_d_reg = zeros(1,length(x)-1);
x_rec_d_reg(1:2:end) = cd;
x_rec_d_reg = conv(x_rec_d_reg,b_hpf,'full');
x_rec_reg = x_rec_a_reg - x_rec_d_reg;


err_dwt_iwt = max(abs(X-x));
err_filter = max(abs(X_rec-x));
err_reg = max(abs(x_rec_reg-x));
