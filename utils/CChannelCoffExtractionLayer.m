%%
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

classdef CChannelCoffExtractionLayer < nnet.layer.Layer
    
    properties
        N %image dim
        S %total number of elements in input matrix
        C %index of interesting color channel
        E %index of interesting element
        %Example:
        % CChannelCoffExtractionLayer('myLayer',201,9,2,1) will return a 
        %201 x 201 matrix with repeated element (2,1) of the reshaped (3x3)
        %matrix of the input. The input contains 9x1 elements.
    end
    
    properties (Learnable)

    end
    
    methods
        function layer = CChannelCoffExtractionLayer(name,N,S,C,E)
            layer.Name = name;
            layer.Description = "Color Channel Coefficients Extraction Layer";
            layer.N = N;
            layer.S = S;
            layer.C = C;
            layer.E = E;
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/layer.S;
            X = reshape(X,[sqrt(layer.S),sqrt(layer.S),L]);
            Z = ones(layer.N,layer.N,1,L,'like',X);
            for i  = 1 : L
                temp = X(layer.C,layer.E,i);
                Z(:,:,:,i) = Z(:,:,:,i) .* temp;
                
            end
            clear X L
        end
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdZ(isnan(dLdZ)) = 0;
            L = length(X(:))/layer.S;
            sz = size(X);
            X = reshape(X,[sqrt(layer.S),sqrt(layer.S),L]);
            dLdZ = reshape(dLdZ,[layer.N,layer.N,L]);
            dLdX = zeros(sqrt(layer.S),sqrt(layer.S),L,'like',X);
            for i = 1 : L
                temp = dLdZ(:,:,i);
                dLdX(layer.C,layer.E,i) = sum(temp(:));
                
            end
            dLdX = reshape(dLdX,sz);
            clear X L dLdZ Z
        end
    end
end