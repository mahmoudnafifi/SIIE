%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2019 Mahmoud Afifi
% York University, Canada
% Email: mafifi@eecs.yorku.ca - m.3afifi@gmail.com
% Permission is hereby granted, free of charge, to any person obtaining 
% a copy of this software and associated documentation files (the 
% "Software"), to deal in the Software with restriction for its use for 
% research purpose only, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
%
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

            Y = squeeze(Y)';
            T = squeeze(T)';
            t_norm =sqrt(sum(T.^2,2));
            y_norm = sqrt(sum(Y.^2,2));
            cosSim=sum(Y.*T,2)./(y_norm.*t_norm + 10^-4);
            cosSim(cosSim>1)=1;
            angles=180/pi * acos_(cosSim);
            
            loss = mean(angles); %final loss
            
        end
        
        function dLdY = backwardLoss(layer, Y, T)

        end
    end
end