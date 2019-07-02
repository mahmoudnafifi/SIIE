classdef imdoubleLayer_ < nnet.layer.Layer
    
    properties
        inType
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = imdoubleLayer_(name, inType)
            layer.Name = name;
            layer.inType = inType;
            if strcmp(layer.inType,'uint8') == 0 && ...
                    strcmp(layer.inType,'uint16') == 0
                error('Wrong inType value in imdoubleLayer_');
            end
            layer.Description = "Layer to convert uint8/uint16 images to double";
        end
        
        function Z = predict(layer, X)
            switch layer.inType
                case 'uint8'
                    Z = X./225;
                case 'uint16'
                    Z = X./65535;
            end
            clear X
        end
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            dLdZ(isnan(dLdZ)) = 0;
            switch layer.inType
                case 'uint8'
                    dLdX = repmat(1/255,size(X)) .* dLdZ;
                case 'uint16'
                    dLdX = repmat(1/65535,size(X)) .* dLdZ;
            end
            clear X dLdZ Z
        end
    end
end