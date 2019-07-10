%% addSplittLayer
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


function layer = addSplittLayer(Name,N,L)
%N: number of channels
%L: channel to be extracted
layer = convolution2dLayer(1,1,'WeightLearnRateFactor',...
    0,'BiasLearnRateFactor',0,'WeightL2Factor',0,'BiasL2Factor',0,...
    'Name',Name);
layer.Weights = zeros(1,1,N,1);
layer.Weights(1,1,L,1) = 1;
layer.Bias = zeros(1,1,1);
end

