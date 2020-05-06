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

classdef reshapeLayer < nnet.layer.Layer
    
    properties
        
    end
    
    properties (Learnable)
        
    end
    
    methods
        function layer = reshapeLayer(name)
            layer.Name = name;
            layer.Description = "Reshape Layer";
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/ 18;
            Z = reshape(X,[1,1,18,L]);
            clear L X
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdX = reshape(dLdZ,size(X));
            clear X dLdZ Z 
        end
        
    end
end