function [] = debug_raw(w)

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
cv_t1222 = w{1}{2}{2}{2};
cd_t1223 = w{1}{2}{2}{3};

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
cv_t2222 = w{2}{2}{2}{2};
cd_t2223 = w{2}{2}{2}{3};

lo11 = w{3}{1}{1};
lo12 = w{3}{1}{2};
lo21 = w{3}{2}{1};
lo22 = w{3}{2}{2};

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/lo11_matlab.hex',lo11,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/lo12_matlab.hex',lo12,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/lo21_matlab.hex',lo21,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/lo22_matlab.hex',lo22,'uint8');

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/ch_t1111_matlab.hex',ch_t1111,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cv_t1112_matlab.hex',cv_t1112,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cd_t1113_matlab.hex',cd_t1113,'uint8');

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/ch_t1121_matlab.hex',ch_t1121,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cv_t1122_matlab.hex',cv_t1122,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cd_t1123_matlab.hex',cd_t1123,'uint8');

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/ch_t1211_matlab.hex',ch_t1211,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cv_t1212_matlab.hex',cv_t1212,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cd_t1213_matlab.hex',cd_t1213,'uint8');


writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/ch_t1221_matlab.hex',ch_t1221,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cv_t1222_matlab.hex',cv_t1222,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cd_t1223_matlab.hex',cd_t1223,'uint8');

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/ch_t2111_matlab.hex',ch_t2111,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cv_t2112_matlab.hex',cv_t2112,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cd_t2113_matlab.hex',cd_t2113,'uint8');

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/ch_t2121_matlab.hex',ch_t2121,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cv_t2122_matlab.hex',cv_t2122,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cd_t2123_matlab.hex',cd_t2123,'uint8');

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/ch_t2211_matlab.hex',ch_t2211,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cv_t2212_matlab.hex',cv_t2212,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cd_t2213_matlab.hex',cd_t2213,'uint8');

writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/ch_t2221_matlab.hex',ch_t2221,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cv_t2222_matlab.hex',cv_t2222,'uint8');
writeRaw('G:\My Drive\Colab Notebooks\MWCNN\output/cd_t2223_matlab.hex',cd_t2223,'uint8');

end