close all
clear

x = 1:256;      % Test signal
J = 2;                 % number of stages
[Faf, Fsf] = FSfarras; % 1st stage anal. & synth. filters
[af, sf] = dualfilt1;
w = dualtree(x, J, Faf, af); 
y = idualtree(w, J, Fsf, sf);

FS_LoD_t1 = Faf{1}(:,1);
FS_HiD_t1 = Faf{1}(:,2);
FS_LoD_t2 = Faf{2}(:,1);
FS_HiD_t2 = Faf{2}(:,2);
LoD_t1 = af{1}(:,1);
HiD_t1 = af{1}(:,2);
LoD_t2 = af{2}(:,1);
HiD_t2 = af{2}(:,2);

FS_LoR_t1 = Fsf{1}(:,1);
FS_HiR_t1 = Fsf{1}(:,2);
FS_LoR_t2 = Fsf{2}(:,1);
FS_HiR_t2 = Fsf{2}(:,2);
LoR_t1 = sf{1}(:,1);
HiR_t1 = sf{1}(:,2);
LoR_t2 = sf{2}(:,1);
HiR_t2 = sf{2}(:,2);

w_my = mydualtree(x, J, FS_LoD_t1, FS_HiD_t1,FS_LoD_t2, FS_HiD_t2, LoD_t1, HiD_t1, LoD_t2, HiD_t2);
y_my = myidualtree(w_my, J, FS_LoR_t1, FS_HiR_t1,FS_LoR_t2, FS_HiR_t2, LoR_t1, HiR_t1, LoR_t2, HiR_t2);

err = max(y_my - y);    
err_w11 = max(w{1}{1} - w_my{1}{1});
err_w12 = max(w{1}{2} - w_my{1}{2});
err_w21 = max(w{2}{1} - w_my{2}{1});
err_w22 = max(w{2}{2} - w_my{2}{2});
