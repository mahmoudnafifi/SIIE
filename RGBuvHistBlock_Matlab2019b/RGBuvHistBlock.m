classdef RGBuvHistBlock < nnet.layer.Layer
    
    properties
        h;
        in;
        C_;
        sigma_;
        learnable;
    end
    
    properties (Learnable)
        C;
        sigma;
        
    end
    
    methods
        function layer = RGBuvHistBlock(name,in,h,C,sigma,learnable)
            layer.Name = name;
            layer.h = h;
            layer.in = in;
            
            layer.sigma = sigma;
            if nargin == 6
                layer.learnable = learnable;
            else
                layer.learnable = 1;
            end
            if layer.learnable == 1
                layer.C = C;
            else
                layer.C_ = C;
            end
            layer.Description = "RGB-uv Histogram Block";
        end
        
        
        
        function Z = predict(layer, X)
            
            eps_= 6.4/layer.h;
            A=[-3.2:eps_:3.19];
            L = length(X(:))/(layer.in * layer.in * 3);
            Z = zeros(layer.h,layer.h,3,L,'like',X);
            for l = 1 : L
                I = X(:,:,:,l);
                I=(reshape(I,[],3));
                hist=zeros(layer.h,layer.h,3,'like',X);
                Iy=sqrt(I(:,1).^2+I(:,2).^2+I(:,3).^2);
                for i = 1 : 3
                    r = setdiff([1,2,3],i);
                    Iu=log(abs(I(:,i))+eps) - log(abs(I(:,r(1)))+eps);
                    Iv=log(abs(I(:,i))+eps) - log(abs(I(:,r(2)))+eps);
                    diff_u=abs(Iu-A);
                    diff_v=abs(Iv-A);
                    if layer.learnable == 1
                        diff_u=exp(-(reshape((reshape(diff_u,[],1).^2),[],size(A,2))/(layer.sigma.^2+eps)));
                        diff_v=exp(-(reshape((reshape(diff_v,[],1).^2),[],size(A,2))/(layer.sigma.^2+eps)));
                    else
                        diff_u=(reshape((reshape(diff_u,[],1)<=eps_/2),[],size(A,2)));
                        diff_v=(reshape((reshape(diff_v,[],1)<=eps_/2),[],size(A,2)));
                    end
                    
                    hist(:,:,i)=(Iy.*double(diff_u))'*double(diff_v);
                    if layer.learnable == 1
                        hist(:,:,i)=sqrt(hist(:,:,i)/sum(sum(hist(:,:,i))));
                    else
                        hist(:,:,i)=sqrt(hist(:,:,i)/(size(I,1)));
                    end
                end
                if layer.learnable == 1
                    Z(:,:,:,l) = hist .* layer.C;
                else
                    Z(:,:,:,l) = hist .* layer.C_;
                end
                
                
            end
            
        end
    end
end
    
    