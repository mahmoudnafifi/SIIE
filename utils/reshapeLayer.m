classdef reshapeLayer < nnet.layer.Layer
    
    properties
        
    end
    
    properties (Learnable)
        
    end
    
    methods
        function layer = reshapeLayer(name)
            layer.Name = name;
            layer.Description = "Reshape Layer";
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/ 18;
            Z = reshape(X,[1,1,18,L]);
            clear L X
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdX = reshape(dLdZ,size(X));
            clear X dLdZ Z 
        end
        
    end
end