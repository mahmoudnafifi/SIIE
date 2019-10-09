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

classdef MatrixNormalizationLayer < nnet.layer.Layer
    
    properties
        N
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = MatrixNormalizationLayer(name,N)
            layer.Name = name;
            layer.Description = "Matrix Normalization Layer";
            layer.N = N;
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/(layer.N*layer.N);
            sz = size(X);
            X = reshape(X,[layer.N*layer.N,L]);
            for i = 1 : L
                n = norm(reshape(X(:,i),[3,3]),1) + 0.0001;
                X(:,i) = X(:,i) ./ n;
            end
            Z = reshape(X,sz);
            clear L X sz n
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            L = length(X(:))/(layer.N*layer.N);
            sz = size(X);
            X = reshape(X,[layer.N*layer.N,L]);
            dLdX = zeros(size(X),'like',X);
            dLdZ = reshape(dLdZ,[layer.N*layer.N,L]);
            for i = 1 : L
                t0 = norm(reshape(X(:,i),[3,3]),1)+0.001;
                dLdX(:,i) = ((1/t0 * eye(layer.N * layer.N) - ...
                    1/t0^3 * X(:,i) *X(:,i)') * dLdZ(:,i))';
            end
            dLdX = reshape(dLdX,sz);
            clear X dLdZ Z L t0 t1 
        end
        
    end
end