%demo
image_name = 'Canon1DsMkIII_0095.png'; %image name 
%Note: be sure that black level subtraction and saturation level-based normalization are applied to the image before using it

model_name = 'trained_model_wo_NUS_Canon1DsMkIII'; %trained model name
in_img_sz = 150; %our network accepts 150x150 raw-RGB image
load(fullfile('models',model_name)); %load the trained model
I = imresize(imread(image_name),[in_img_sz,in_img_sz]); %resize the image
est_ill = predict(trained_model,I); %estimate the scene illuminant
est_ill =  est_ill./norm(est_ill); %make it a unit vector
fprintf('Estimated scene illuminant =  %f, %f, %f\n',...
    est_ill(1),est_ill(2),est_ill(3)); %display the result

subplot(1,2,1); imshow(I*4); title('input raw-RGB img'); %show input raw-RGB image (here, we scale it by 4 to aid visualization)
subplot(1,2,2); imshow(reshape(...
    reshape(im2double(I),[],3)*diag(est_ill(2)./est_ill),size(I))*4);  %apply white balance correction then show the result (scaled to aid visualization)
title('White-balanced raw-RGB img');

