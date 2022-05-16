clc;
clear all;
warning('off', 'Images:initSize:adjustingMag');

% qp_input = 40;
% video_black = sprintf('decoded_%i_g.yuv', qp_input);
input = 'cropped_sasha.yuv';

%create black video
frames=120;
% fileID = fopen(video_black,'w');
fw=128;
fh=128;
% tempY=zeros(1,fw*fh);
% tempUV=128*ones(1,fw*fh/2);
% for f=1:frames
%      fwrite(fileID,tempY,'uint8');
%      fwrite(fileID,tempUV,'uint8');
% end
% fclose(fileID);

%create sound video
%comment it and just use your sound video file name soundvideo
% if 1
%     fileID = fopen(input,'r');
%     tempY = fread(fileID,fw*fh,'uint8');
%     tempUV = fread(fileID,fw*fh/2,'uint8');
%     fclose(fileID);
%     
%     fileID = fopen(soundvideo,'w');
%     for f=1:frames
%          fwrite(fileID,tempY,'uint8');
%          fwrite(fileID,tempUV,'uint8');
%     end
%     fclose(fileID);
% end

%encoding without 
% qp = [15,20,30,40]; % РК 
% qp = [15]; %ЗК
qp = [32, 34, 36, 40]; 

if 1
    copyfile(input,'input.yuv');
    for i=1:length(qp)
        video_black = sprintf('cropped_decoded_%i_g.yuv', qp(i));
        copyfile(video_black, 'soundvideo.yuv');

        %Encode via 3D-HEVC
        s = sprintf('"./mvhm/3dhevc/TAppEncoderStatic" -c "./mvhm/3dhevc/baseCfg_2view.cfg" -c "./mvhm/3dhevc/seqCfg.cfg" -b "./mvhm/3dhevc/video.bin" -q %i -f %i',qp(i),frames);
        system(s);
        %Decode via 3D-HEVC 
        s = sprintf('"./mvhm/3dhevc/TAppDecoderStatic" -b "./mvhm/3dhevc/video.bin" -o "./input_dec.yuv"');
        system(s);

        s = sprintf('C_input_dec_%i.yuv',i);
        
        copyfile('input_dec_1.yuv',s);
        
        PSNR = PSNRfromFile(input, s, fw, fh, frames);
%         results = sprintf('results/generated_%i/results.txt', qp_input);
        results = sprintf('results.txt');
        fileID = fopen(results,'at');
        fprintf(fileID,'Q=%i PSNR=%g\n',qp(i),PSNR);
        fclose(fileID);
           
%         bitrate = sprintf('bitrate.txt')
%         copyfile('bitrate.txt', bitrate);
    end
end

% copyfile(soundvideo,'soundvideo.yuv');
% copyfile(input,'input.yuv');
% for i=1:length(qp)
%     %Encode via 3D-HEVC
%     s = sprintf('"./mvhm/3dhevc/TAppEncoderStatic" -c "./mvhm/3dhevc/baseCfg_2view.cfg" -c "./mvhm/3dhevc/seqCfg.cfg" -b "./mvhm/3dhevc/video.bin" -q %i -f %i',qp(i),frames);
%     system(s);
%     %Decode via 3D-HEVC 
%     s = sprintf('"./mvhm/3dhevc/TAppDecoderStatic" -b "./mvhm/3dhevc/video.bin" -o "./input_dec.yuv"');
%     system(s);
%     s = sprintf('input_dec_Sound%i.yuv',i);
%     copyfile('input_dec_1.yuv',s);
%     
%     PSNR = PSNRfromFile(input, s, fw,fh,frames);
%     fileID = fopen('results.txt','at');
%     fprintf(fileID,'WithSound: Q=%i PSNR=%g\n',qp(i),PSNR);
%     fclose(fileID);
% end




