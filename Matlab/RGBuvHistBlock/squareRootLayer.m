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

classdef squareRootLayer < nnet.layer.Layer

    properties
    end

    properties (Learnable)
        
    end
    
    methods
        function layer = squareRootLayer(name)
           layer.Name = name;
           layer.Description = "Square Root Layer";
           
        end
        
        function Z = predict(layer, X)
            Z = sqrt(abs(X));
            clear X
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdZ(isnan(dLdZ)) = 0;
           dLdX = 1/2 .* abs(X)./X .* abs(X).^(-0.5) .* dLdZ;
           clear X dLdZ Z
        end
    end
end