clear

[af, sf] = farras;     % analysis and synthesis filters


for i = 1:32
x=zeros(1,64);         % create zero signal 
w = mydwt(x,3,af(:,1),af(:,2));       % analysis filter banks (3 stages)
w{1}(i)=1; % set single coefficient to 1
y = myidwt(w,3,sf(:,1),sf(:,2));      % synthesis filter banks (3 stages)
plot(0:63,y);          % Plot the wavelet
axis([0 63 -0.5 0.5]);     
title('Standard 1-D wavelet') 
xlabel('t');                    
ylabel('\psi(t)');
saveas(gcf, ["output/w_1_" + num2str(i) + ".png"]);
end

for i = 1:16
x=zeros(1,64);         % create zero signal 
w = mydwt(x,3,af(:,1),af(:,2));       % analysis filter banks (3 stages)
w{2}(i)=1; % set single coefficient to 1
y = myidwt(w,3,sf(:,1),sf(:,2));      % synthesis filter banks (3 stages)
plot(0:63,y);          % Plot the wavelet
axis([0 63 -0.5 0.5]);     
title('Standard 1-D wavelet') 
xlabel('t');                    
ylabel('\psi(t)');
saveas(gcf, ["output/w_2_" + num2str(i) + ".png"]);
end

for i = 1:8
x=zeros(1,64);         % create zero signal 
w = mydwt(x,3,af(:,1),af(:,2));       % analysis filter banks (3 stages)
w{3}(i)=1; % set single coefficient to 1
y = myidwt(w,3,sf(:,1),sf(:,2));      % synthesis filter banks (3 stages)
plot(0:63,y);          % Plot the wavelet
axis([0 63 -0.5 0.5]);     
title('Standard 1-D wavelet') 
xlabel('t');                    
ylabel('\psi(t)');
saveas(gcf, ["output/w_3_" + num2str(i) + ".png"]);
end

for i = 1:8
x=zeros(1,64);         % create zero signal 
w = mydwt(x,3,af(:,1),af(:,2));       % analysis filter banks (3 stages)
w{4}(i)=1; % set single coefficient to 1
y = myidwt(w,3,sf(:,1),sf(:,2));      % synthesis filter banks (3 stages)
plot(0:63,y);          % Plot the wavelet
axis([0 63 -0.5 0.5]);     
title('Standard 1-D wavelet') 
xlabel('t');                    
ylabel('\psi(t)');
saveas(gcf, ["output/w_4_" + num2str(i) + ".png"]);
end