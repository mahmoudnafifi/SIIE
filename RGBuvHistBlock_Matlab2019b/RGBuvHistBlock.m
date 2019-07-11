%Create a histogram RGB-uv block to plugin it into a CNN model. There are
%two options: 1) a histogram block with 2 learnable parameters that control 
%the histogram generation process and 2) a histogram block without 
%learnable parameters.

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
% 1- Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown. When Color
% Constancy Goes Wrong: Correcting Improperly White-Balanced Images. 
% In CVPR, 2019
% 2- Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Examples
%%%%%%%%%%%%
%%1- Create RGB-uv histogram block with leranable parameters (C & sigma) (Ref: Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019)
%a = -0.2;
%b = 0.2;
%r = (b-a).*rand(1,1) + a;
%inputImageSize = 151;
%histogramOutSize = 61;
%C = [r r r] + 1 ; %scale
%sigma = [r r r] + 0.5; %fall-off factor
%learnable = 1; %histBlock with learnable parameters
%histBlock = RGBuvHistBlock('HistBlock',inputImageSize,histogramOutSize,C,sigma,learnable); 

%%2- Create RGB-uv histogram block without leranable parameters (Ref: Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown. When Color
% Constancy Goes Wrong: Correcting Improperly White-Balanced Images. In CVPR, 2019)
%C = 100; %static scale factor
%sigma = []; %fall-off factor
%learnable = 0; %no learnable parameters
%inputImageSize = 151;
%histogramOutSize = 61;
%histBlock = RGBuvHistBlock('HistBlock',inputImageSize,histogramOutSize,C,sigma,learnable); 

%%add the created histogram block to your network
%inputLayer = imageInputLayer([151 151 3],'Name','input','Normalization','none');
%conv1 = convolution2dLayer(3,64,'Name','conv1','Stride',[2,2]);
%relu1 = reluLayer('Name','relu1');
%.....
%outLayer = regressionLayer('Name','outLayer');
%Layers = [inputLayer
%       histBlock
%       conv1
%       relu1
%       ...
%       outLayer];

%%      
classdef RGBuvHistBlock < nnet.layer.Layer
    
    properties
        h; %output histogram dimension (e.g., h=61 will create 61x61x3 hist)
        in; %input image dimension (e.g., in=151 means input to this layer should be 151x151x3)
        C_; %if you chose non-learnable histogram, the final histogram will be scaled by the value of C_
        learnable; %learnable = 1 to choose the histogram block with learnable parameters.
    end
    
    properties (Learnable)
        C; %if with learnable param, C is the learnable scale vector
        sigma; %sigma is the learnable fall-off factor vector
    end
    
    methods
        function layer = RGBuvHistBlock(name,in,h,C,sigma,learnable)
            layer.Name = name;
            layer.h = h;
            layer.in = in;
            
            layer.sigma = sigma;
            if nargin == 6
                layer.learnable = learnable;
            else
                layer.learnable = 1;
            end
            if layer.learnable == 1
                layer.C = C;
            else
                layer.C_ = C;
            end
            layer.Description = "RGB-uv Histogram Block";
        end
        
        
        
        function Z = predict(layer, X)
            eps_= 6.4/layer.h;
            A=[-3.2:eps_:3.19];
            L = length(X(:))/(layer.in * layer.in * 3);
            Z = zeros(layer.h,layer.h,3,L,'like',X);
            for l = 1 : L
                I = X(:,:,:,l);
                I=(reshape(I,[],3));
                hist=zeros(layer.h,layer.h,3,'like',X);
                Iy=sqrt(I(:,1).^2+I(:,2).^2+I(:,3).^2);
                for i = 1 : 3
                    r = setdiff([1,2,3],i);
                    Iu=log(abs(I(:,i))+eps) - log(abs(I(:,r(1)))+eps);
                    Iv=log(abs(I(:,i))+eps) - log(abs(I(:,r(2)))+eps);
                    diff_u=abs(Iu-A);
                    diff_v=abs(Iv-A);
                    if layer.learnable == 1
                        diff_u=exp(-(reshape((reshape(diff_u,[],1).^2),[],size(A,2))/(layer.sigma(i).^2+eps)));
                        diff_v=exp(-(reshape((reshape(diff_v,[],1).^2),[],size(A,2))/(layer.sigma(i).^2+eps)));
                    else
                        diff_u=(reshape((reshape(diff_u,[],1)<=eps_/2),[],size(A,2)));
                        diff_v=(reshape((reshape(diff_v,[],1)<=eps_/2),[],size(A,2)));
                    end     
                    hist(:,:,i)=(Iy.*double(diff_u))'*double(diff_v);
                    if layer.learnable == 1
                        hist(:,:,i)=sqrt(hist(:,:,i)) .* layer.C(i);
                    else
                        hist(:,:,i)=sqrt(hist(:,:,i)/(size(I,1)));
                    end
                end
                if layer.learnable == 0
                    Z(:,:,:,l) = hist .* layer.C_;
                else
                    Z(:,:,:,l)= hist;
                end
            end
        end
    end
end
    
    