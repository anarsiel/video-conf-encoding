clc;
clear all;
warning('off', 'Images:initSize:adjustingMag');

% PSNR = PSNRfromFile(input, s, fw, fh, frames);

% only lips
% R1 = [5.2117, 4.8267, 4.465, 3.8733];
% PSNR1 = [35.0333, 34.0392, 32.8005, 30.3341];
% R2 = [5.2083, 4.8367, 4.4783, 3.7867];
% PSNR2 = [35.1678, 34.1669, 33.0173, 30.4198];


% only face
R1 = [13.1767, 11.395, 9.915, 7.7617];
PSNR1 = [35.2243, 34.0497, 32.915, 30.4153];
R2 = [13.2017, 11.4717, 9.945, 7.7717];
PSNR2 = [35.4901, 34.3655, 33.2399, 30.8257];

dRate = bjontegaard2(R1,PSNR1,R2,PSNR2,'rate')
dPSNR = bjontegaard2(R1,PSNR1,R2,PSNR2,'dsnr')