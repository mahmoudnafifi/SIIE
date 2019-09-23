%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2019-present, Mahmoud Afifi
% York University, Canada
% Email: mafifi@eecs.yorku.ca - m.3afifi@gmail.com
% All rights reserved.
%
%%
% Please cite the following work if this program is used:
% Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination
% Estimation for DNN Models. In BMVC, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
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
            clear cos_angles t y L
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            %%Input:
            % Y: Predicted vectors (1,1,3,L) -- each prediction is a 1x3
            % vector
            % T: Ground truth vectors (1,1,3,L) -- each ground truth sample
            % is a 1x3 vector
            % -Note: L is the number of mini-batches
            %%Output:
            % dLdY: derivative of the loss with respect to predicted
            % vectors in Y
            %%
            L = length(T(:))/3; %get number of mini-batches
            dLdY = zeros(1,1,3,L,'like',Y); %derivative of loss (output)
            for i = 1 : L %for each mini-batch, do
                y = reshape(Y(:,:,:,i),[1,3]); %get current prediction
                t = reshape(T(:,:,:,i),[1,3]); %get current ground-truth
                y(isnan(y)) = 0; t(isnan(t)) =0;
                if sum(y(:)) == 0
                    y = y + eps*10^-5;
                end
                if sum(t(:)) == 0
                    t = t + eps*10^-5;
                end
                y_norm = norm(y); %L2norm of y (vector length)
                t_norm = norm(t); %L2norm of t (vector length)
                a = (1/(y_norm * t_norm) * t - 1/(y_norm^3 * t_norm) * ...
                    sum(y.*t) * y)'; %derivative of cos similarity w.r.t. y
                b = sum(y.*t).^2/(norm(y).^2*norm(t).^2); %cos similarity^2
                dLdY(1,1,:,i) = -a/sqrt(1-b) * 180/(L*pi); %derivative of
                % arccos(cos_simialrity(y,y))
            end
            clear a b L y_norm t_norm
        end
    end
end