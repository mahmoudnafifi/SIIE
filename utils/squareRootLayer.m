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