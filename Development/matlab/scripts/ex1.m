clear

teta = linspace(0,1,256);


lpf = exp(j*(teta./2)).*cos(teta/2);
hpf = -j*exp(j*(teta./2)).*sin(teta/2);


figure(1)
plot (teta,real(lpf))
xlabel('teta')
ylabel('H(\Theta)')
set(get(gca,'ylabel'),'rotation',0)
figure(2)
plot (teta,real(hpf))
ylabel('G(\Theta)')
xlabel('teta')
set(get(gca,'ylabel'),'rotation',0)
