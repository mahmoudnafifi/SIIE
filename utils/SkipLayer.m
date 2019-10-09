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

classdef SkipLayer < nnet.layer.Layer
    
    properties
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = SkipLayer(name)
            layer.Name = name;
            %bypass layer
            layer.Description = "SkipLayer";
            
        end
        
        function Z = predict(layer, X)
            Z = X;
            if sum(X(:)) ~=67500
            imwrite(gather(X),'mapped.png');
            end
            clear X
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdX = dLdZ;
            clear X dLdZ Z
        end
        
    end
end