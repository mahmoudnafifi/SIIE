classdef SkipLayer < nnet.layer.Layer
    
    properties
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = SkipLayer(name)
            layer.Name = name;
            %bypass layer
            layer.Description = "SkipLayer";
            
        end
        
        function Z = predict(layer, X)
            Z = X;
            clear X
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdX = dLdZ;
            clear X dLdZ Z
        end
        
    end
end