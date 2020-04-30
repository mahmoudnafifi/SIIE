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
input_dir = fullfile('..','imgs_w_normalization'); %input image directory

ext = '.png';

output_dir = fullfile('..','imgs_w_normalization_results'); %output directory

Matlab_ver = '2018b'; %'2018b', '2019a', or 'higher'

if exist(output_dir,'dir') == 0
    mkdir(output_dir);
end

images = dir(fullfile(input_dir,['*' ext]));
images = {images(:).name};
%Note: be sure that the image is in the raw-RGB linear space and the
%black/saturation normalization is correctly applied to the image before
%using it.
est_ills = zeros(length(images),3);
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
for i =  1 : length(images)
    fprintf('Processing (%d/%d) ... \n',i, length(images));
    image_name = fullfile(input_dir,images{i});
    
    I_ = imread(image_name); %resize the image
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
    est_ills(i,:) = reshape(est_ill,1,3);
end

save(fullfile(output_dir,'results.mat'),'est_ills','-v7.3');