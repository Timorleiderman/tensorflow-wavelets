
close all
clear
addpath DTCWT/2D 
addpath utils

x = single(imread('../input/LennaGrey.png'));

J = 2;
[Faf, Fsf] = FSfarras;
[af, sf] = dualfilt1;

w = mycplxdual2D(x, J, Faf, af);
ch_t1111 = w{1}{1}{1}{1};
cv_t1112 = w{1}{1}{1}{2};
cd_t1113 = w{1}{1}{1}{3};

ch_t1121 = w{1}{1}{2}{1};
cv_t1122 = w{1}{1}{2}{2};
cd_t1123 = w{1}{1}{2}{3};

ch_t1211 = w{1}{2}{1}{1};
cv_t1212 = w{1}{2}{1}{2};
cd_t1213 = w{1}{2}{1}{3};

ch_t1221 = w{1}{2}{2}{1};
cv_t1221 = w{1}{2}{2}{2};
cd_t1221 = w{1}{2}{2}{3};

ch_t2111 = w{2}{1}{1}{1};
cv_t2112 = w{2}{1}{1}{2};
cd_t2113 = w{2}{1}{1}{3};

ch_t2121 = w{2}{1}{2}{1};
cv_t2122 = w{2}{1}{2}{2};
cd_t2123 = w{2}{1}{2}{3};

ch_t2211 = w{2}{2}{1}{1};
cv_t2212 = w{2}{2}{1}{2};
cd_t2213 = w{2}{2}{1}{3};

ch_t2221 = w{2}{2}{2}{1};
cv_t2221 = w{2}{2}{2}{2};
cd_t2221 = w{2}{2}{2}{3};

lo11 = w{3}{1}{1};
lo12 = w{3}{1}{2};
lo21 = w{3}{2}{1};
lo22 = w{3}{2}{2};

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/matlab_lo11.hex',lo11,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/matlab_lo12.hex',lo12,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/matlab_lo21.hex',lo21,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/matlab_lo22.hex',lo22,'uint8');

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/matlab_ch_t2121.hex',ch_t2121,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/matlab_cv_t2122.hex',cv_t2122,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/matlab_cd_t2123.hex',cd_t2123,'uint8');


y = myicplxdual2D(w, J, Fsf, sf);


err = max(max(abs(y - x))); 
