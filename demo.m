%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2019-present, Mahmoud Afifi
% York University, Canada
% Email: mafifi@eecs.yorku.ca - m.3afifi@gmail.com
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
% All rights reserved.
%
%%
% Please cite the following work if this program is used:
% Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

Matlab_ver = 'higher'; %'2018b', '2019a', or 'higher'
image_name = fullfile('imgs_w_normalization',...
    'Cube+_challenge_CanonEOS550D_243.png'); %image name 
%Note: be sure that the image is in the raw-RGB linear space and the 
%black/saturation normalization is correctly applied to the image before 
%using it.
model_name = 'trained_model_wo_NUS_Canon1DsMkIII'; %trained model name
device = 'gpu'; %cpu
in_img_sz = 150; %our network accepts 150x150 raw-RGB image
if strcmpi(Matlab_ver, '2018b') == 1 || strcmpi(Matlab_ver,'2019a') == 1
    old = 1;
    load(fullfile('models_old',model_name)); %load the trained model
else
    old = 0;
    load(fullfile('models',model_name)); %load the trained model
end
I_ = imread(image_name); %read the image
sz =size(I_);
if sz(1)~=in_img_sz || sz(2)~=in_img_sz
    I = imresize(I_,[in_img_sz,in_img_sz]); %resize the image
else
    I = I_;
end
%estimate the scene illuminant
if old == 1
    est_ill = predict(trained_model,I,'ExecutionEnvironment',device); 
else
    est_ill = predict_(trained_model,I,device); %estimate the scene illuminant
end
est_ill =  est_ill./norm(est_ill); %make it a unit vector
fprintf('Estimated scene illuminant =  %f, %f, %f\n',...
    est_ill(1),est_ill(2),est_ill(3)); %display the result

factor = 6; %scale factor to aid visualization

subplot(1,3,1); imshow(I_*factor); 
title('Input raw-RGB image'); %show input raw-RGB image (scaled to aid visualization)
subplot(1,3,2); imshow(imresize(imread('mapped.png')*factor,[sz(1) sz(2)]));
title('mapped image');
subplot(1,3,3); imshow(reshape(...
    reshape(im2double(I_),[],3)*diag(est_ill(2)./est_ill),sz)*factor);  %apply white balance correction then show the result (scaled to aid visualization)
title('White-balanced raw-RGB image');

linkaxes