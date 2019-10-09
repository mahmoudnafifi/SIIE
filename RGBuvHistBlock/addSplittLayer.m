%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2019-present, Mahmoud Afifi
% York University, Canada
% Email: mafifi@eecs.yorku.ca - m.3afifi@gmail.com
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
% All rights reserved.
%
%%
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

