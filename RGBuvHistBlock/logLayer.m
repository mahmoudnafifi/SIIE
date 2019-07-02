classdef logLayer < nnet.layer.Layer

    properties
       
    end

    properties (Learnable)
       
    end
    
    methods
        function layer = logLayer(name)
           layer.Name = name;
           layer.Description = "Log layer";
           
        end
        
        function Z = predict(layer, X)
            Z = log(abs(X)+eps);
            clear X
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
           dLdZ(isnan(dLdZ)) = 0;
           dLdX = 1./(X+eps) .* dLdZ;
           clear X dLdZ Z
        end
    end
end