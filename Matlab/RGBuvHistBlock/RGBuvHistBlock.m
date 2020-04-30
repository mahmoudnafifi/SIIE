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
% 1- Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown. When Color
% Constancy Goes Wrong: Correcting Improperly White-Balanced Images. 
% In CVPR, 2019
% 2- Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

%% %%%%%%%%%%%%%%%%%%%%%%%
% Matlab 2019b or higher %
%% %%%%%%%%%%%%%%%%%%%%%%%
 
 
%Create a histogram RGB-uv block to plugin it into a CNN model. There are
%two options: 1) a histogram block with 2 learnable parameters that control 
%the histogram generation process and 2) a histogram block without 
%learnable parameters.


%% Examples
%%%%%%%%%%%%
%%1- Create RGB-uv histogram block with leranable parameters (C & sigma) (Ref: Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019)

%r = rand(1,1) * 0.02;
%inputImageSize = 151;
%histogramOutSize = 61;
%C = [r r r] + 1 ; %scale
%sigma_u = [r r r] + 0.05; %fall-off factor (u)
%sigma_v = [r r r] + 0.05; %fall-off factor (v)
%learnable = 1; %histBlock with learnable parameters
%histBlock = RGBuvHistBlock('HistBlock',inputImageSize,histogramOutSize,C,sigma_u,sigma_v,learnable); 

%%2- Create RGB-uv histogram block without leranable parameters (Ref: Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown. When Color
% Constancy Goes Wrong: Correcting Improperly White-Balanced Images. In CVPR, 2019)
%C = 100; %static scale factor
%sigma_u = []; %fall-off factor (u)
%sigma_v = []; %fall-off factor (v)
%learnable = 0; %no learnable parameters
%inputImageSize = 151;
%histogramOutSize = 61;
%histBlock = RGBuvHistBlock('HistBlock',inputImageSize,histogramOutSize,C,sigma_u,sigma_v,learnable); 

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
classdef RGBuvHistBlock < nnet.layer.Layer %Matlab 2019b or higher
    
    properties
        h; %output histogram dimension (e.g., h=61 will create 61x61x3 hist)
        in; %input image dimension (e.g., in=151 means input to this layer should be 151x151x3)
        C_; %if you chose non-learnable histogram, the final histogram will be scaled by the value of C_
        A;
        eps_;
        learnable; %learnable = 1 to choose the histogram block with learnable parameters.
    end
    
    properties (Learnable)
        C; %if with learnable param, C is the learnable scale vector
        sigma_u; %sigma is the learnable fall-off factor vector
        sigma_v; %sigma is the learnable fall-off factor vector
    end
    
    methods
        function layer = RGBuvHistBlock(name,in,h,C,sigma_u,sigma_v,learnable)
            if nargin == 0
                name = 'RGB-UV-Histogram-Block';
                in = 100;
                h = 61;
                C = [1 1 1];
                sigma_u = [];
                sigma_v = [];
                learnable = 0;
            elseif nargin == 1
                in = 100;
                h = 61;
                C = [1 1 1];
                sigma_u = [];
                sigma_v = [];
                learnable = 0;
            elseif nargin == 2
                h = 61;
                C = [1 1 1];
                sigma_u = [];
                sigma_v = [];
                learnable = 0;
            elseif nargin == 3
                C = [1 1 1];
                sigma_u = [];
                sigma_v = [];
                learnable = 0;
            elseif nargin == 4
                sigma_u = [];
                sigma_v = [];
                learnable = 0;
            elseif nargin == 5
                sigma_v = [];
                learnable = 0;
            elseif nargin == 6
                learnable = 0;
            end
            
            layer.Name = name;
            layer.h = h;
            layer.in = in;
            layer.eps_= 6.4/layer.h;
            layer.A = [-3.2:layer.eps_:3.19];
            layer.learnable = learnable;
            if layer.learnable == 1
                layer.C = C;
                layer.sigma_u = sigma_u;
                layer.sigma_v = sigma_v;
            else
                layer.sigma_u = [];
                layer.sigma_v = [];
                layer.C_ = C;
            end
            layer.Description = "RGB-uv Histogram Block";
        end
        
        
        
        function Z = predict(layer, X)
            sz = size(X);
            if sz(1) ~= layer.in && sz(2) ~= layer.in
                if sz(1) ~= layer.in
                    inds_1 = round(linspace(1,size(X,1),layer.in));
                else
                    inds_1 = linspace(1,sz(1));
                end
                if sz(2) ~= layer.in
                    inds_2 = round(linspace(1,size(X,2),layer.in));
                else
                    inds_2 = linspace(1,sz(2));
                end
                if length(sz) == 3
                    X = X(inds_1,inds_2,:);
                else
                    X = X(inds_1,inds_2,:,:);
                end
            end
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
                    diff_u=abs(Iu-layer.A);
                    diff_v=abs(Iv-layer.A);
                    if layer.learnable == 1
                        diff_u=exp(-(reshape((reshape(diff_u,[],1).^2),...
                            [],size(layer.A,2))/(layer.sigma_u(i).^2+eps)));
                        diff_v=exp(-(reshape((reshape(diff_v,[],1).^2),...
                            [],size(layer.A,2))/(layer.sigma_v(i).^2+eps)));
                    else
                        diff_u=(reshape((reshape(diff_u,[],1)<=...
                            layer.eps_/2),[],size(layer.A,2)));
                        diff_v=(reshape((reshape(diff_v,[],1)<=...
                            layer.eps_/2),[],size(layer.A,2)));
                    end     
                    hist(:,:,i)=(Iy.*double(diff_u))'*double(diff_v);
                    if layer.learnable == 1
                        hist(:,:,i)= sqrt(hist(:,:,i)) .* layer.C(i);
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
    
    