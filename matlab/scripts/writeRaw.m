% write binery file from image

function [] = writeRaw(file_path,img,dtype)

fileID = fopen(file_path,'w');
fwrite(fileID,img',dtype);
fclose(fileID);


end
    