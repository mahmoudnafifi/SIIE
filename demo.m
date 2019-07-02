%demo
image_name = fullfile('imgs','GehlerShi_Canon5d_IMG_0883.png'); %image name 
%Note: be sure that the image is in the sraw-RGB linear space and the 
%black/saturation normalization are correctly applied to the image before 
%using it.

model_name = 'trained_model_wo_NUS_Canon1DsMkIII'; %trained model name
%model_name = 'trained_model_wo_NUS_Canon1DsMkIII'; %trained model name
in_img_sz = 150; %our network accepts 150x150 raw-RGB image
load(fullfile('models',model_name)); %load the trained model
I = imread(image_name); %resize the image
sz =size(I);
if sz(1)~=in_img_sz || sz(2)~=in_img_sz
    I = imresize(I,[in_img_sz,in_img_sz]); %resize the image
end
est_ill = predict(trained_model,I); %estimate the scene illuminant
est_ill =  est_ill./norm(est_ill); %make it a unit vector
fprintf('Estimated scene illuminant =  %f, %f, %f\n',...
    est_ill(1),est_ill(2),est_ill(3)); %display the result

subplot(1,3,1); imshow(I*4); title('Input raw-RGB image'); %show input raw-RGB image (here, we scale it by 4 to aid visualization)
subplot(1,3,2); imshow(imread('mapped.png')*4); title('mapped image');
subplot(1,3,3); imshow(reshape(...
    reshape(im2double(I),[],3)*diag(est_ill(2)./est_ill),size(I))*4);  %apply white balance correction then show the result (scaled to aid visualization)
title('White-balanced raw-RGB image');

