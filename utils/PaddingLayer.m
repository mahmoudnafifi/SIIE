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

classdef PaddingLayer < nnet.layer.Layer

    properties
        N
        T
    end

    properties (Learnable)
      
    end
    
    methods
        function layer = PaddingLayer(name,N,T)
           layer.Name = name;
           layer.Description = "Padding Layer";
           layer.N = N;
           layer.T = T;
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/layer.N;
            X = reshape(X,[3,L]);
            temp = zeros(layer.T - layer.N,1,'like',X);
            Z = zeros(1,1,layer.T,L,'like',X);
            for i = 1 : L
                Z(:,:,:,i) = [X(:,i); temp];
            end
            clear X L temp
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
           dLdZ(isnan(dLdZ)) = 0;
           sz = size(X);
           L = length(X(:))/layer.N;
           dLdX = zeros(1,1,layer.N,L,'like',X);
           for i = 1  : L
                dLdX(:,:,:,i) = dLdZ(:,:,1:3,i);
           end
           dLdX = reshape(dLdX,sz);
           clear X dLdZ Z sz
        end
    end
end