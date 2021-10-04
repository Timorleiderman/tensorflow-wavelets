close all
clear

kronDelta1 = zeros(128,1);
kronDelta1(60) = 1;
kronDelta2 = zeros(128,1);
kronDelta2(64) = 1;

origmode = dwtmode('status','nodisplay');
dwtmode('per','nodisp')
J = 3;
[dwt1C,dwt1L] = wavedec(kronDelta1,J,'sym7');
[dwt2C,dwt2L] = wavedec(kronDelta2,J,'sym7');
dwt1Cfs = detcoef(dwt1C,dwt1L,3);
dwt2Cfs = detcoef(dwt2C,dwt2L,3);

[dt1A,dt1D] = dualtree(kronDelta1,'Level',J,'FilterLength',14);
[dt2A,dt2D] = dualtree(kronDelta2,'Level',J,'FilterLength',14);
dt1Cfs = dt1D{3};
dt2Cfs = dt2D{3};

figure(1)
subplot(1,2,1)
stem(abs(dwt1Cfs),'markerfacecolor',[0 0 1])
title({'DWT';['Squared 2-norm = ' num2str(norm(dwt1Cfs,2)^2,3)]},'fontsize',10)
ylim([0 0.4])
subplot(1,2,2)
stem(abs(dwt2Cfs),'markerfacecolor',[0 0 1])
title({'DWT';['Squared 2-norm = ' num2str(norm(dwt2Cfs,2)^2,3)]},'fontsize',10)
ylim([0 0.4])

figure(2)
subplot(1,2,1)
stem(abs(dt1Cfs),'markerfacecolor',[0 0 1])
title({'Dual-tree CWT';['Squared 2-norm = ' num2str(norm(dt1Cfs,2)^2,3)]},'fontsize',10)
ylim([0 0.4])
subplot(1,2,2)
stem(abs(dwt2Cfs),'markerfacecolor',[0 0 1])
title({'Dual-tree CWT';['Squared 2-norm = ' num2str(norm(dt2Cfs,2)^2,3)]},'fontsize',10)
ylim([0 0.4])

load wecg
dt = 1/180;
t = 0:dt:(length(wecg)*dt)-dt;
figure(3)
plot(t,wecg)
xlabel('Seconds')
ylabel('Millivolts')

figure(4)
J = 6; 
[df,rf] = dtfilters('farras');
[dtDWT1,L1] = wavedec(wecg,J,df(:,1),df(:,2));
details = zeros(2048,3);
details(2:4:end,2) = detcoef(dtDWT1,L1,2);
details(4:8:end,3) = detcoef(dtDWT1,L1,3);
subplot(3,1,1)
stem(t,details(:,2),'Marker','none','ShowBaseline','off')
title('Level 2')
ylabel('mV')
subplot(3,1,2)
stem(t,details(:,3),'Marker','none','ShowBaseline','off')
title('Level 3')
ylabel('mV')
subplot(3,1,3)
plot(t,wecg)
title('Original Signal')
xlabel('Seconds')
ylabel('mV')

figure(5)
[dtcplxA,dtcplxD] = dualtree(wecg,'Level',J,'FilterLength',14);
details = zeros(2048,3);
details(2:4:end,2) = dtcplxD{2};
details(4:8:end,3) = dtcplxD{3};
subplot(3,1,1)
stem(t,real(details(:,2)),'Marker','none','ShowBaseline','off')
title('Level 2')
ylabel('mV')
subplot(3,1,2)
stem(t,real(details(:,3)),'Marker','none','ShowBaseline','off')
title('Level 3')
ylabel('mV')
subplot(3,1,3)
plot(t,wecg)
title('Original Signal')
xlabel('Seconds')
ylabel('mV')



wecgShift = circshift(wecg,4);
[dtDWT2,L2] = wavedec(wecgShift,J,df(:,1),df(:,2));

detCfs1 = detcoef(dtDWT1,L1,1:J,'cells');
apxCfs1 = appcoef(dtDWT1,L1,rf(:,1),rf(:,2),J);
cfs1 = horzcat(detCfs1,{apxCfs1});
detCfs2 = detcoef(dtDWT2,L2,1:J,'cells');
apxCfs2 = appcoef(dtDWT2,L2,rf(:,1),rf(:,2),J);
cfs2 = horzcat(detCfs2,{apxCfs2});

sigenrgy = norm(wecg,2)^2;
enr1 = cell2mat(cellfun(@(x)(norm(x,2)^2/sigenrgy)*100,cfs1,'uni',0));
enr2 = cell2mat(cellfun(@(x)(norm(x,2)^2/sigenrgy)*100,cfs2,'uni',0));
levels = {'D1';'D2';'D3';'D4';'D5';'D6';'A6'};
enr1 = enr1(:);
enr2 = enr2(:);
table(levels,enr1,enr2,'VariableNames',{'Level','enr1','enr2'})