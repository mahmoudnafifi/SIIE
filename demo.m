%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2019 Mahmoud Afifi
% York University, Canada
% Email: mafifi@eecs.yorku.ca - m.3afifi@gmail.com
% Permission is hereby granted, free of charge, to any person obtaining 
% a copy of this software and associated documentation files (the 
% "Software"), to deal in the Software with restriction for its use for 
% research purpose only, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
%
% Please cite the following work if this program is used:
% Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

image_name = fullfile('imgs','NUS_Canon1DsMkIII_0095.png'); %image name 
%Note: be sure that the image is in the sraw-RGB linear space and the 
%black/saturation normalization are correctly applied to the image before 
%using it.

model_name = 'trained_model_wo_NUS_Canon1DsMkIII'; %trained model name
in_img_sz = 150; %our network accepts 150x150 raw-RGB image
load(fullfile('models',model_name)); %load the trained model
I_ = imread(image_name); %resize the image
sz =size(I_);
if sz(1)~=in_img_sz || sz(2)~=in_img_sz
    I = imresize(I_,[in_img_sz,in_img_sz]); %resize the image
else
    I = I_;
end
est_ill = predict(trained_model,I); %estimate the scene illuminant
est_ill =  est_ill./norm(est_ill); %make it a unit vector
fprintf('Estimated scene illuminant =  %f, %f, %f\n',...
    est_ill(1),est_ill(2),est_ill(3)); %display the result

subplot(1,3,1); imshow(I_*4); title('Input raw-RGB image'); %show input raw-RGB image (here, we scale it by 4 to aid visualization)
subplot(1,3,2); imshow(imread('mapped.png')*4); title('mapped image');
subplot(1,3,3); imshow(reshape(...
    reshape(im2double(I_),[],3)*diag(est_ill(2)./est_ill),size(I))*4);  %apply white balance correction then show the result (scaled to aid visualization)
title('White-balanced raw-RGB image');

