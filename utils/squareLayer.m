classdef squareLayer < nnet.layer.Layer

    properties
    end

    properties (Learnable)
    end
    
    methods
        function layer = squareLayer(name)
           layer.Name = name;
           layer.Description = "Square Layer";
           
        end
        
        function Z = predict(layer, X)
            Z = X.^2;
            clear X
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
           dLdZ(isnan(dLdZ)) = 0;
           dLdX = 2 .* X .* dLdZ;
           clear X dLdZ Z
        end
    end
end