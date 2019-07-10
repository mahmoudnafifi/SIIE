%% PaddingLayer
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

classdef PaddingLayer < nnet.layer.Layer

    properties
        N
        T
    end

    properties (Learnable)
      
    end
    
    methods
        function layer = PaddingLayer(name,N,T)
           layer.Name = name;
           layer.Description = "Padding Layer";
           layer.N = N;
           layer.T = T;
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/layer.N;
            X = reshape(X,[3,L]);
            temp = zeros(layer.T - layer.N,1,'like',X);
            Z = zeros(1,1,layer.T,L,'like',X);
            for i = 1 : L
                Z(:,:,:,i) = [X(:,i); temp];
            end
            clear X L temp
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
           dLdZ(isnan(dLdZ)) = 0;
           sz = size(X);
           L = length(X(:))/layer.N;
           dLdX = zeros(1,1,layer.N,L,'like',X);
           for i = 1  : L
                dLdX(:,:,:,i) = dLdZ(:,:,1:3,i);
           end
           dLdX = reshape(dLdX,sz);
           clear X dLdZ Z sz
        end
    end
end