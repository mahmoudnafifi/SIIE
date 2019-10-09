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

classdef absLayer < nnet.layer.Layer
    
    properties
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = absLayer(name)
            layer.Name = name;
            layer.Description = "Absolute Layer";
        end
        

        
        function Z = predict(layer, X)
            X = real(X);
            X(isnan(X)) = 0;
            Z = abs(X);
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            X = real(X);
            X(isnan(X)) = 0;
            dLdX = X./abs(X);
        end
        
    end
end