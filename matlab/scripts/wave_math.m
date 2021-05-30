
clear
clear all
close all

NUM_OF_SAMPLES = 128;
t_start = -2*pi;
t_end = 2*pi;
t = linspace(t_start,t_end,NUM_OF_SAMPLES);

A=0.5;

y1 = A*cos(2*pi*100*t);
y2 = A*sin(2*pi*t);
y3 = A*cos(2*pi*20*t);
y4 = A*sin(2*pi*8*t);
y5 = A*sin(2*pi*16*t);

y_cat = [y1, y2, y3, y4, y5];
t_cat = linspace(t_start,t_end,(length(y_cat)));

sigma = 5;
figure(1);

filename = 'mooving_window.gif';
del = 0.1; % time between animation frames
k = 1;
for idx = -3.14 : 0.1: 9.43
      mu = -pi+idx;
      gauss = (1/sigma*sqrt(2*pi))*exp(-((t_cat-mu).^2)/2*sigma^2);
      h = plot(t_cat, y_cat,t_cat,gauss);
      set(h,{'LineWidth'},{1;5});
      set(gca,'xtick',[]);
      set(gca,'ytick',[]);
      title('fliter width problem');
      xlabel('time') ;
      ylabel('input signal') ;
      drawnow;
      frame = getframe(1);
      im = frame2im(frame);
      [imind,cm] = rgb2ind(im,256);
      if k == 1
        [imind, cm] = rgb2ind(im,256);
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf,'DelayTime',del);
      else
        imind = rgb2ind(im, cm);
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append','DelayTime',del);
      end
    k = k+1;
      

end

