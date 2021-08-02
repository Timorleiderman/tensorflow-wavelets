close all
clear

x = rand(256,128);
J = 2;
[Faf, Fsf] = FSfarras;
[af, sf] = dualfilt1;
w = dualtree2D(x, J, Faf, af);

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

w_my = mydualtree2D(x, J, FS_LoD_t1, FS_HiD_t1,FS_LoD_t2, FS_HiD_t2, LoD_t1, HiD_t1, LoD_t2, HiD_t2);


y = idualtree2D(w_my, J, Fsf, sf);

err = x - y; 

err_w111 = max(max(w{1}{1}{1} - w_my{1}{1}{1}));
err_w112 = max(max(w{1}{1}{2} - w_my{1}{1}{2}));
err_w121 = max(max(w{1}{2}{1} - w_my{1}{2}{1}));
err_w122 = max(max(w{1}{2}{2} - w_my{1}{2}{2}));
err_w211 = max(max(w{2}{1}{1} - w_my{2}{1}{1}));
err_w212 = max(max(w{2}{1}{2} - w_my{2}{1}{2}));
err_w221 = max(max(w{2}{2}{1} - w_my{2}{2}{1}));
err_w222 = max(max(w{2}{2}{2} - w_my{2}{2}{2}));

err_w31 = max(max(w{3}{1} - w_my{3}{1}));

max(max(abs(err)))