classdef imdoubleLayer < nnet.layer.Layer
    
    properties
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = imdoubleLayer(name)
            layer.Name = name;
            layer.Description = "Layer to convert uint16 images to double";
        end
        
        function Z = predict(layer, X)
            Z = X./65535;
            clear X
        end
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdZ(isnan(dLdZ)) = 0;
            dLdX = repmat(1/65535,size(X)) .* dLdZ;
            clear X dLdZ Z
        end
    end
end