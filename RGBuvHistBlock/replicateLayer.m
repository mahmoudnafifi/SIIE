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

classdef replicateLayer < nnet.layer.Layer

    properties
        N
        h
    end

    properties (Learnable)
       
    end
    
    methods
        function layer = replicateLayer(name,N,h)
           layer.Name = name;
           layer.Description = "Replicate Input Layer";
           layer.N = N;
           layer.h = h;
           
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/(layer.N);
            Z = zeros(sqrt(layer.N),sqrt(layer.N),layer.h,L,'like',X);
            for i = 1 : L
                Z(:,:,:,i) = repmat(X(:,:,i),1,1,1,layer.h);
            end
            clear X L
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
           dLdZ(isnan(dLdZ)) = 0;
           sz = size(X);
           dLdX = reshape(sum(dLdZ,3),sz);
           clear X dLdZ Z
        end
    end
end