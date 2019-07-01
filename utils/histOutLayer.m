classdef histOutLayer < nnet.layer.Layer
    
    properties
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = histOutLayer(name,h)
            layer.Name = name;
            %this layer for debuging .. just pass the output histogram
            layer.Description = "Histogram Output Layer";
            
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