%% squareRootLayer
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