clear

Hd = [0 -1 5 12 5 -1 0 0]/20*sqrt(2);
Hr = [0 -3 -15 73 170 73 -15 -3]/280*sqrt(2);

Gd = [0 3 -15 -73 170 -73 -15 3]/280*sqrt(2);
Gr = [0 -1 -5 12 -5 -1 0 0]/20*sqrt(2);

% analysis (decomposition)
fbAna = dwtfilterbank('Wavelet','Custom','CustomScalingFilter',[Hd' Hr'],'CustomWaveletFilter',[Gd' Gr']);
% synthesis (reconstruction)
fbSyn = dwtfilterbank('Wavelet','Custom','CustomScalingFilter',[Hd' Hr'],'CustomWaveletFilter',[Gd' Gr'],'FilterType','Synthesis');


[fbAna_phi,t] = scalingfunctions(fbAna);
[fbAna_psi,~] = wavelets(fbAna);
[fbSyn_phi,~] = scalingfunctions(fbSyn);
[fbSyn_psi,~] = wavelets(fbSyn);
subplot(2,2,1)
plot(t,fbAna_phi(end,:))
grid on
title('Analysis - Scaling')
subplot(2,2,2)
plot(t,fbAna_psi(end,:))
grid on
title('Analysis - Wavelet')
subplot(2,2,3)
plot(t,fbSyn_phi(end,:))
grid on
title('Synthesis - Scaling')
subplot(2,2,4)
plot(t,fbSyn_psi(end,:))
grid on
title('Synthesis - Wavelet')