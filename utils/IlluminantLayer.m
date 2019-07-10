%% IlluminantLayer
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

classdef IlluminantLayer < nnet.layer.Layer
    
    properties
        
    end
    
    properties (Learnable)
        
    end
    
    methods
        function layer = IlluminantLayer(name)
            layer.Name = name;
            layer.Description = "Final Illuminant Calculation Layer";
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/ 18;
            X = reshape(X,[9,2,L]);
            Z = zeros(1,1,3,L,'like',X);
            for i = 1 : L
                Z(1,1,:,i) = reshape(sum(reshape(X(:,1,i),[3,3]) * ...
                    reshape(X(:,2,i),[3,3]),2),[1 1 3]);
            end
            clear L X
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdZ(isnan(dLdZ)) = 0;
            L = length(X(:))/ 18;
            sz = size(X);
            dLdX = zeros(1,1,9,2,L,'like',X);
            dLdZ = reshape(dLdZ,[1,3,L]);
            X = reshape(X,[9,2,L]);
            O = ones(3,1,'like',X);
            for i = 1  : L 
              temp  = dLdZ(1,:,i)' * (reshape(X(:,2,i),[3,3]) * O)';
              dLdX(1,1,:,1,i) = reshape(temp,[1 1 9]);
              temp  = reshape(X(:,1,i),[3,3])' * dLdZ(1,:,i)' * O';
              dLdX(1,1,:,2,i) = reshape(temp,[1 1 9]);
            end
            dLdX = reshape(dLdX,sz);
            clear X dLdZ Z sz L temp
        end
        
    end
end