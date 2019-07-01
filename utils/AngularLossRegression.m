classdef AngularLossRegression < nnet.layer.RegressionLayer


    properties
    end
 
    methods
        function layer = AngularLossRegression(name)
            layer.Name = name;
            layer.Description = "Angular Error Loss Layer";
        end

        function loss = forwardLoss(layer, Y, T)
            %%Input:
            % Y: Predicted vectors (1,1,3,L) -- each prediction is a 1x3
            % vector
            % T: Ground truth vectors (1,1,3,L) -- each ground truth sample
            % is a 1x3 vector
            % -Note: L is the number of mini-batches
            %%Output:
            % loss: angular error between each vector in Y and T
            %%
            L = length(T(:))/3; %Get number of mini-batches
            cos_angles = zeros(1,1,1,L,'like',Y); % cos(angles) between 
            % vectors in Y and T
            
            for i = 1 : L %For each mini-batch, do
                t = reshape(T(:,:,:,i),[3,1]); %current ground truth vector
                y = reshape(Y(:,:,:,i),[3,1]); %current prediction
                cos_angles (i) = sum(y.*t)/(norm(t) * norm(y)); %compute 
                %cosine similarity between y and y
            end
            
            cos_angles(cos_angles>1)=1; %trim any cos(angle) > 1
            loss = mean(180/pi * acos(cos_angles)); %final loss
        end
        
        function dLdY = backwardLoss(layer, Y, T)

        end
    end
end