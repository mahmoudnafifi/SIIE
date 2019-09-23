%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2019-present, Mahmoud Afifi
% York University, Canada
% Email: mafifi@eecs.yorku.ca - m.3afifi@gmail.com
% All rights reserved.
%
%%
% Please cite the following work if this program is used:
% Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%



classdef ExponentialKernelLayer < nnet.layer.Layer
    
    properties
        h;
        eps;
        N;
    end
    
    properties (Learnable)
        sigma;
    end
    
    methods
        function layer = ExponentialKernelLayer(name,N,h)
            layer.N = N;
            layer.Name = name;
            layer.Description = "Exponential Kernel Layer";
            layer.h = h;
            layer.eps= 6.4/layer.h;
            layer.sigma = rand * 0.2;
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/layer.N;
            X=(reshape(X,layer.N,L));
            A=repmat([-3.2:layer.eps:3.19],size(X,1),1);
            Z = zeros(layer.N,layer.h,L,'like',X);
            for i = 1 : L
                Z(:,:,i) = exp(-(X(:,i) - A).^2./layer.sigma.^2);
            end
            Z = reshape(Z,[sqrt(layer.N),sqrt(layer.N),layer.h,L]);
            clear A X
        end
        
        
        
        function [dLdX,dLdsigma] = backward(layer, X, Z, dLdZ, memory)
            
            dLdZ(isnan(dLdZ)) = 0;
            M = length(dLdZ(:))/(layer.N*layer.h);
            X=(reshape(X,[],M));
            dLdZ = reshape(dLdZ,[layer.N,layer.h,M]);
            A=[-3.2:layer.eps:3.19];
            dLdX = zeros(layer.N,M,'like',X);
            for i = 1 : M
                dLdX(:,i) = sum(-2 .* (X(:,i) - A)/layer.sigma.^2 ...
                    .* exp(-(X(:,i) - A).^2./layer.sigma.^2) .* ...
                    dLdZ(:,:,i),2);
            end
            
            L = length(dLdX(:))/(layer.N);
            if L > 1
                dLdX = reshape(dLdX,[sqrt(layer.N),sqrt(layer.N),1,L]);
            else
                dLdX = reshape(dLdX,[sqrt(layer.N),sqrt(layer.N),L]);
            end
            dLdsigma = zeros(1,M,'like',X);
            for  i = 1 : M
                dLdsigma(:,i) = sum(sum(2 .* (X(:,i) - A).^2 /layer.sigma.^3 ...
                    .* exp(-(X(:,i) - A).^2./layer.sigma.^2) .*dLdZ(:,:,i)));
            end
            dLdsigma = sum(dLdsigma);
            clear A dLdZ X
        end
        
    end
end