%% imdoubleLayer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2019 Mahmoud Afifi
% York University, Canada
% Email: mafifi@eecs.yorku.ca - m.3afifi@gmail.com
% Permission is hereby granted, free of charge, to any person obtaining 
% a copy of this software and associated documentation files (the 
% "Software"), to deal in the Software with restriction for its use for 
% research purpose only, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
%
% Please cite the following work if this program is used:
% Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

classdef imdoubleLayer < nnet.layer.Layer
    
    properties
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = imdoubleLayer(name)
            layer.Name = name;
            layer.Description = "Layer to convert uint16 images to double";
        end
        
        function Z = predict(layer, X)
            Z = X./65535; %we assume X is always uint16
            clear X
        end
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdZ(isnan(dLdZ)) = 0;
            dLdX = repmat(1/65535,size(X)) .* dLdZ;
            clear X dLdZ Z
        end
    end
end