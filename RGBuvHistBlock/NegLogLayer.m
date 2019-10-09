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

classdef NegLogLayer < nnet.layer.Layer

    properties
        
    end

    properties (Learnable)
        
    end
    
    methods
        function layer = NegLogLayer(name)
           layer.Name = name;
           layer.Description = "Negative Log layer";
           
        end
        
        function Z = predict(layer, X)
            Z = -log(abs(X)+eps);
            clear X
            
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdZ(isnan(dLdZ)) = 0;
           dLdX = -1 * 1./(X+eps) .* dLdZ;
           clear X Z dLdZ
        end
    end
end