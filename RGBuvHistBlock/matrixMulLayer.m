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

classdef matrixMulLayer < nnet.layer.Layer

    properties
        h;
        N;
    end

    properties (Learnable)
       
    end
    
    methods
        function layer = matrixMulLayer(name,N,h)
           layer.N = N;
           layer.Name = name;
           layer.Description = "Matrix Multiplication Layer";
           layer.h = h;
           
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/(layer.N*layer.h*2);
            X = reshape(X,[layer.N,layer.h,2,L]);
            Z = zeros(layer.h,layer.h,'like',X);
            for i = 1 : L
                Z(:,:,i) = reshape(X(:,:,1,i)' * X(:,:,2,i),...
                    [layer.h,layer.h]);
            end
            clear X
            
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
             L = length(X(:))/(layer.N*layer.h*2);
             dLdZ(isnan(dLdZ)) = 0;
             dLdX = zeros(sqrt(layer.N),sqrt(layer.N),layer.h,2,L,'like',X);
             X = reshape(X,[layer.N,layer.h,2,L]);
             dLdZ = reshape(dLdZ,layer.h,layer.h,L);
             for i = 1 : L
                dLdX(:,:,:,1,i) = reshape(X(:,:,2,i) * dLdZ(:,:,i)', ...
                    sqrt(layer.N),sqrt(layer.N),layer.h,1);
                dLdX(:,:,:,2,i) = reshape(X(:,:,1,i) * dLdZ(:,:,i), ...
                      sqrt(layer.N),sqrt(layer.N),layer.h,1);
             end
             dLdX = reshape(dLdX,[sqrt(layer.N),sqrt(layer.N),layer.h*2,L]);
             clear dLdZ Z X
        end
    end
end