function PSNR = PSNRfromFile(fname1,fname2,fw,fh,frames) 

fileID1 = fopen(fname1,'rb');
fileID2 = fopen(fname2,'rb');
Ydec  = [];
mse = 0;
for f=1:(frames)
    Y1 = fread(fileID1,fw*fh,'uint8');
    UV1 = fread(fileID1,fw*fh/2,'uint8');
    Y2 = fread(fileID2,fw*fh,'uint8');
    UV2 = fread(fileID2,fw*fh/2,'uint8');
    Y1 = double(Y1);
    Y2 = double(Y2);
    D = Y1-Y2;
    D=mean(D(:).*D(:));
    mse = mse + D;
end
fclose(fileID1);
fclose(fileID2);
mse = mse/(frames);
PSNR = 10*log10(255*255/mse);
