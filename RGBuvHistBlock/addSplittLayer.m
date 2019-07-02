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

