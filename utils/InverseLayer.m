classdef InverseLayer < nnet.layer.Layer

    properties
        R %random noise
        N
    end

    properties (Learnable)
    end
    
    methods
        function layer = InverseLayer(name,N)
           layer.Name = name;
           layer.Description = "Inverse Layer";
           layer.N = N;
           layer.R = rand(sqrt(layer.N),sqrt(layer.N))/1000;
           
        end
        
        function Z = predict(layer, X)
            L = length(X(:))/layer.N;
            sz = size(X);
            X = reshape(X,[sqrt(layer.N),sqrt(layer.N),L]);
            Z = zeros(sqrt(layer.N),sqrt(layer.N),'like',X);
            for i = 1 : L
                if det(X(:,:,i)) == 0
                    Z(:,:,i) = reshape(pinv(X(:,:,i) + layer.R),...
                        [sqrt(layer.N),sqrt(layer.N)]);
                else
                    Z(:,:,i) = reshape(pinv(X(:,:,i)),...
                        [sqrt(layer.N),sqrt(layer.N)]);
                end
            end
            Z = reshape(Z,sz);
            clear X L sz
        end

        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
           dLdZ(isnan(dLdZ)) = 0;
           L = length(X(:))/layer.N;
           sz = size(X);
           X = reshape(X,[sqrt(layer.N),sqrt(layer.N),L]); 
           dLdZ = reshape(dLdZ,[sqrt(layer.N),sqrt(layer.N),L]); 
           dLdX = zeros(sqrt(layer.N),sqrt(layer.N),L,'like',X);
           for i = 1 : L
               if det(X(:,:,i))==0
                    invX = pinv(X(:,:,i) + layer.R);
               else
                   invX = pinv(X(:,:,i));
               end
               dLdX(:,:,i) = - reshape(reshape(dLdZ(:,:,i),1,[]) * ...
                   kron(invX',invX),...
                   [sqrt(layer.N),sqrt(layer.N)]);
           end
           dLdX = reshape(dLdX,sz);
           clear X dLdZ Z L sz
        end
    end
end