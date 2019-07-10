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
% Mahmoud Afifi and Michael S. Brown, Sensor Independent Illumination 
% Estimation for DNN Models, BMVC, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

classdef scaleLayer < nnet.layer.Layer
    
    properties
        h
        N
    end
    
    properties (Learnable)
        C
    end
    
    methods
        function layer = scaleLayer(name,N,h)
            layer.Name = name;
            layer.Description = "Histogram Normalization Layer";
            layer.h = h;
            layer.N = N;
            layer.C = rand * sqrt(1/layer.N);
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/(layer.h*layer.h);
            Z = reshape(X.*layer.C,[layer.h,layer.h,1,L]);
            clear L X
        end
        
        
        
        function [dLdX,dLdC] = backward(layer, X, Z, dLdZ, memory)
            dLdZ(isnan(dLdZ)) = 0;
            dLdX = reshape(layer.C .* dLdZ,size(X));
            dLdC = X .* dLdZ;
            dLdC = sum(dLdC(:));
            clear X dLdZ Z
        end
        
    end
end