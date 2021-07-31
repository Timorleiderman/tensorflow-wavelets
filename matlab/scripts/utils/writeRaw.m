% write binery file from image
% littel endien

function [] = writeRaw(file_path,img,dtype)

fileID = fopen(file_path,'w');
fwrite(fileID,img',dtype, 'l');
fclose(fileID);


end
    