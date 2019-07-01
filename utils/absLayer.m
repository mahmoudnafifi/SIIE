classdef absLayer < nnet.layer.Layer
    
    properties
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = absLayer(name)
            layer.Name = name;
            layer.Description = "Absolute Layer";
        end
        

        
        function Z = predict(layer, X)
            X = real(X);
            X(isnan(X)) = 0;
            L = length(X(:))/9;
            Z = zeros(size(X),'like',X);
            for i = 1 : L
                Z(:,:,:,i) = abs(X(:,:,:,i)); 
            end
            clear L
        end
        
        
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            X = real(X);
            X(isnan(X)) = 0;
            L = length(X(:))/9;
            dLdX = zeros(size(X),'like',X);
            
            for i = 1 : L
                X(:,:,:,i) = X(:,:,:,i);
                dLdX(:,:,:,i) = X(:,:,:,i)./abs(X(:,:,:,i)).* ...
                    dLdZ(:,:,:,i);
            end
            clear X dLdZ Z L
        end
        
    end
end