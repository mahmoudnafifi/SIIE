%% matrixHadamardProdLayer
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

classdef matrixHadamardProdLayer < nnet.layer.Layer
    
    properties
        h;
        N;
    end
    
    properties (Learnable)
        
    end
    
    methods
        function layer = matrixHadamardProdLayer(name,N,h)
            layer.N = N;
            layer.Name = name;
            layer.Description = "Matrix Hadamard (entry-wise) Product Layer";
            layer.h = h;
            
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/(layer.N * (2 * layer.h));
            Z = zeros(sqrt(layer.N),sqrt(layer.N),layer.h,L,'like',X);
            for i = 1 : L
                Z(:,:,:,i) = X(:,:,1:layer.h,i) .* ...
                    X(:,:,layer.h+1:end,i);
            end
            clear X
        end
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
           
            L = length(X(:))/(layer.N*(layer.h*2));
            dLdZ(isnan(dLdZ)) = 0;
            dLdX = zeros(size(X),'like',X);
            for i = 1 : L
                dLdX(:,:,1:layer.h,i) = dLdZ(:,:,:,i) .* ...
                    X(:,:,layer.h+1:end,i);
                dLdX(:,:,layer.h+1:end,i) = dLdZ(:,:,:,i) .* ...
                    X(:,:,1:layer.h,i);
            end
            clear X Z dLdZ
        end
    end

end
