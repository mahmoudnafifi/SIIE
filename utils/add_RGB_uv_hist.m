function [lgraph,lastLayer] = add_RGB_uv_hist(inputLayer,...
    InputLayerName,prefix,InputSize,histSize,lgraph,firstLayerLgraph)

if nargin == 4
    histSize = 61;
    lgraph = [];
    firstLayerLgraph = [];
elseif nargin == 5
    lgraph = [];
    firstLayerLgraph = [];
elseif nargin == 6
    firstLayerLgraph = [];
elseif nargin ~=7
    error('Not enough input arguments');
end

%% Splitter layers
%Red
SplittRedChannelLayer = addSplittLayer ([prefix ...
    'Red Channel Splitter Layer'],3,1);
%Green
SplittGreenChannelLayer = addSplittLayer ([prefix ...
    'Green Channel Splitter Layer'],3,2);
%Blue
SplittBlueChannelLayer = addSplittLayer ([prefix ...
    'Blue Channel Splitter Layer'],3,3);


%% Log layers
R_LogLayer = logLayer([prefix 'R Log Layer']);
G_LogLayer = logLayer([prefix 'G Log Layer']);
B_LogLayer = logLayer([prefix 'B Log Layer']);

%% Negative Log Layers
R_NegLogLayer = NegLogLayer([prefix 'R Negative Log Layer']);
G_NegLogLayer = NegLogLayer([prefix 'G Negative Log Layer']);
B_NegLogLayer = NegLogLayer([prefix 'B Negative Log Layer']);


%% Square layers
R_squareLayer = squareLayer([prefix 'R Square Layer']);
G_squareLayer = squareLayer([prefix 'G Square Layer']);
B_squareLayer = squareLayer([prefix 'B Square Layer']);

%% Square root layer
squareRootLayer_RGB = squareRootLayer([prefix 'Square Root Layer R+G+B']);
squareRootLayer_uv1 = squareRootLayer([prefix 'Square Root Layer uv1']);
squareRootLayer_uv2 = squareRootLayer([prefix 'Square Root Layer uv2']);
squareRootLayer_uv3 = squareRootLayer([prefix 'Square Root Layer uv3']);

%% Replicate channel layer
repLayer = replicateLayer([prefix 'Replicate Channels Layer'],...
    InputSize * InputSize, histSize);

%% Add Layers
addLayer_RGB = additionLayer(3,'Name',[prefix 'Add Layer RGB']);
addLayer_uv1_1 = additionLayer(2,'Name',[prefix 'Add Layer uv1_1']);
addLayer_uv1_2 = additionLayer(2,'Name',[prefix 'Add Layer uv1_2']);
addLayer_uv2_1 = additionLayer(2,'Name',[prefix 'Add Layer uv2_1']);
addLayer_uv2_2 = additionLayer(2,'Name',[prefix 'Add Layer uv2_2']);
addLayer_uv3_1 = additionLayer(2,'Name',[prefix 'Add Layer uv3_1']);
addLayer_uv3_2 = additionLayer(2,'Name',[prefix 'Add Layer uv3_2']);

%% Exponensial Kernel Layers
ExpLayer_uv1_1 = ExponentialKernelLayer([prefix ...
    'Exponensial Layer uv1_1'],InputSize*InputSize,histSize);
ExpLayer_uv1_2 = ExponentialKernelLayer([prefix ...
    'Exponensial Layer uv1_2'],InputSize*InputSize,histSize);
ExpLayer_uv2_1 = ExponentialKernelLayer([prefix ...
    'Exponensial Layer uv2_1'],InputSize*InputSize,histSize);
ExpLayer_uv2_2 = ExponentialKernelLayer([prefix ...
    'Exponensial Layer uv2_2'],InputSize*InputSize,histSize);
ExpLayer_uv3_1 = ExponentialKernelLayer([prefix ...
    'Exponensial Layer uv3_1'],InputSize*InputSize,histSize);
ExpLayer_uv3_2 = ExponentialKernelLayer([prefix ...
    'Exponensial Layer uv3_2'],InputSize*InputSize,histSize);

%% Depth Concatination Layers
catLayer_uv1_1 = depthConcatenationLayer(2,'Name',[prefix ...
    'Concatination Layer1 uv1']);
catLayer_uv2_1 = depthConcatenationLayer(2,'Name',[prefix ...
    'Concatination Layer1 uv2']);
catLayer_uv3_1 = depthConcatenationLayer(2,'Name',[prefix ...
    'Concatination Layer1 uv3']);

catLayer_uv1_2 = depthConcatenationLayer(2,'Name',[prefix ...
    'Concatination Layer2 uv1']);
catLayer_uv2_2 = depthConcatenationLayer(2,'Name',[prefix ...
    'Concatination Layer2 uv2']);
catLayer_uv3_2 = depthConcatenationLayer(2,'Name',[prefix ...
    'Concatination Layer2 uv3']);

catLayer_hist = depthConcatenationLayer(3,'Name',[prefix ...
    'Histogram Concatination Layer']);

%% Hadamard Product Layers
hadamardLayer_uv1 = matrixHadamardProdLayer([prefix ...
    'Hadamard Product Layer uv1'],InputSize * InputSize , histSize);
hadamardLayer_uv2 = matrixHadamardProdLayer([prefix ...
    'Hadamard Product Layer uv2'],InputSize * InputSize , histSize);
hadamardLayer_uv3 = matrixHadamardProdLayer([prefix ...
    'Hadamard Product Layer uv3'],InputSize * InputSize , histSize);

%% Matrix Multiplication Layers
matrixMulLayer_uv1 = matrixMulLayer([prefix ...
    'Matrix Multiplication Layer uv1'],InputSize * InputSize , histSize);
matrixMulLayer_uv2 = matrixMulLayer([prefix ...
    'Matrix Multiplication Layer uv2'],InputSize * InputSize , histSize);
matrixMulLayer_uv3 = matrixMulLayer([prefix ...
    'Matrix Multiplication Layer uv3'],InputSize * InputSize , histSize);

%% Normalization Layer (factorize the histogram by a learnable factor)
normLayer_uv1 = scaleLayer([prefix 'Normalization Layer uv1'], ...
    InputSize * InputSize, histSize);
normLayer_uv2 = scaleLayer([prefix 'Normalization Layer uv2'], ...
    InputSize * InputSize, histSize);
normLayer_uv3 = scaleLayer([prefix 'Normalization Layer uv3'], ...
    InputSize * InputSize, histSize);


%% Histogram Output Layer
HistOutLayer = histOutLayer([prefix 'Histogram Output Layer']);
%% Graph construction
% Adding new layers
layers = [ 
    inputLayer
    SplittRedChannelLayer
    R_squareLayer
    ];

if isempty(lgraph) == 0
    lgraph = addLayers(lgraph,layers);
else
    lgraph = layerGraph(layers);
end

layers = [
    SplittGreenChannelLayer
    G_squareLayer
    ];

lgraph = addLayers(lgraph,layers);

layers = [
    SplittBlueChannelLayer
    B_squareLayer
    ];

lgraph = addLayers(lgraph,layers);

lgraph = addLayers(lgraph,R_LogLayer);

lgraph = addLayers(lgraph,R_NegLogLayer);

lgraph = addLayers(lgraph,G_LogLayer);

lgraph = addLayers(lgraph,G_NegLogLayer);

lgraph = addLayers(lgraph,B_LogLayer);

layers = [
    squareRootLayer_RGB
    repLayer];

lgraph = addLayers(lgraph,layers);

lgraph = addLayers(lgraph,B_NegLogLayer);

lgraph = addLayers(lgraph,addLayer_RGB);

lgraph = addLayers(lgraph,addLayer_uv1_1);

lgraph = addLayers(lgraph,addLayer_uv1_2);

lgraph = addLayers(lgraph,addLayer_uv2_1);

lgraph = addLayers(lgraph,addLayer_uv2_2);

lgraph = addLayers(lgraph,addLayer_uv3_1);

lgraph = addLayers(lgraph,addLayer_uv3_2);

lgraph = addLayers(lgraph,ExpLayer_uv1_1);

lgraph = addLayers(lgraph,ExpLayer_uv1_2);

lgraph = addLayers(lgraph,ExpLayer_uv2_1);

lgraph = addLayers(lgraph,ExpLayer_uv2_2);

lgraph = addLayers(lgraph,ExpLayer_uv3_1);

lgraph = addLayers(lgraph,ExpLayer_uv3_2);

lgraph = addLayers(lgraph,catLayer_uv1_1);

lgraph = addLayers(lgraph,catLayer_uv2_1);

lgraph = addLayers(lgraph,catLayer_uv3_1);

lgraph = addLayers(lgraph,catLayer_uv1_2);

lgraph = addLayers(lgraph,catLayer_uv2_2);

lgraph = addLayers(lgraph,catLayer_uv3_2);

lgraph = addLayers(lgraph,hadamardLayer_uv1);

lgraph = addLayers(lgraph,hadamardLayer_uv2);

lgraph = addLayers(lgraph,hadamardLayer_uv3);

lgraph = addLayers(lgraph,matrixMulLayer_uv1);

lgraph = addLayers(lgraph,matrixMulLayer_uv2);

lgraph = addLayers(lgraph,matrixMulLayer_uv3);

lgraph = addLayers(lgraph,normLayer_uv1);

lgraph = addLayers(lgraph,normLayer_uv2);

lgraph = addLayers(lgraph,normLayer_uv3);

lgraph = addLayers(lgraph,squareRootLayer_uv1);

lgraph = addLayers(lgraph,squareRootLayer_uv2);

lgraph = addLayers(lgraph,squareRootLayer_uv3);

lgraph = addLayers(lgraph,[catLayer_hist;HistOutLayer]);


% Layer connections
lgraph = connectLayers(lgraph,InputLayerName,[prefix ...
    'Green Channel Splitter Layer']);

if isempty(firstLayerLgraph) == 0
    lgraph = connectLayers(lgraph,InputLayerName,firstLayerLgraph);
end

lgraph = connectLayers(lgraph,InputLayerName,...
    [prefix 'Blue Channel Splitter Layer']);

lgraph = connectLayers(lgraph,[prefix 'Red Channel Splitter Layer'],...
    [prefix 'R Negative Log Layer']);

lgraph = connectLayers(lgraph,[prefix 'Red Channel Splitter Layer'],...
    [prefix 'R Log Layer']);

lgraph = connectLayers(lgraph,[prefix 'Green Channel Splitter Layer'],...
    [prefix 'G Negative Log Layer']);

lgraph = connectLayers(lgraph,[prefix 'Green Channel Splitter Layer'],...
    [prefix 'G Log Layer']);

lgraph = connectLayers(lgraph,[prefix 'Blue Channel Splitter Layer'],...
    [prefix 'B Negative Log Layer']);

lgraph = connectLayers(lgraph,[prefix 'Blue Channel Splitter Layer'],...
    [prefix 'B Log Layer']);

lgraph = connectLayers(lgraph,[prefix 'R Square Layer'],...
    [prefix 'Add Layer RGB/in1']);

lgraph = connectLayers(lgraph,[prefix 'G Square Layer'],...
    [prefix 'Add Layer RGB/in2']);

lgraph = connectLayers(lgraph,[prefix 'B Square Layer'],...
    [prefix 'Add Layer RGB/in3']);

lgraph = connectLayers(lgraph,[prefix 'Add Layer RGB'],...
    [prefix 'Square Root Layer R+G+B']);

lgraph = connectLayers(lgraph,[prefix 'R Log Layer'],...
    [prefix 'Add Layer uv1_1/in1']);

lgraph = connectLayers(lgraph,[prefix 'G Negative Log Layer'],...
    [prefix 'Add Layer uv1_1/in2']);

lgraph = connectLayers(lgraph,[prefix 'R Log Layer'],...
    [prefix 'Add Layer uv1_2/in1']);

lgraph = connectLayers(lgraph,[prefix 'B Negative Log Layer'],...
    [prefix 'Add Layer uv1_2/in2']);

lgraph = connectLayers(lgraph,[prefix 'G Log Layer'],...
    [prefix 'Add Layer uv2_1/in1']);

lgraph = connectLayers(lgraph,[prefix 'R Negative Log Layer'],...
    [prefix 'Add Layer uv2_1/in2']);

lgraph = connectLayers(lgraph,[prefix 'G Log Layer'],...
    [prefix 'Add Layer uv2_2/in1']);

lgraph = connectLayers(lgraph,[prefix 'B Negative Log Layer'],...
    [prefix 'Add Layer uv2_2/in2']);

lgraph = connectLayers(lgraph,[prefix 'B Log Layer'],...
    [prefix 'Add Layer uv3_1/in1']);

lgraph = connectLayers(lgraph,[prefix 'R Negative Log Layer'],...
    [prefix 'Add Layer uv3_1/in2']);

lgraph = connectLayers(lgraph,[prefix 'B Log Layer'],...
    [prefix 'Add Layer uv3_2/in1']);

lgraph = connectLayers(lgraph,[prefix 'G Negative Log Layer'],...
    [prefix 'Add Layer uv3_2/in2']);

lgraph = connectLayers(lgraph,[prefix 'Add Layer uv1_1'],...
    [prefix 'Exponensial Layer uv1_1']);

lgraph = connectLayers(lgraph,[prefix 'Add Layer uv1_2'],...
    [prefix 'Exponensial Layer uv1_2']);

lgraph = connectLayers(lgraph,[prefix 'Add Layer uv2_1'],...
    [prefix 'Exponensial Layer uv2_1']);

lgraph = connectLayers(lgraph,[prefix 'Add Layer uv2_2'],...
    [prefix 'Exponensial Layer uv2_2']);

lgraph = connectLayers(lgraph,[prefix 'Add Layer uv3_1'],...
    [prefix 'Exponensial Layer uv3_1']);

lgraph = connectLayers(lgraph,[prefix 'Add Layer uv3_2'],...
    [prefix 'Exponensial Layer uv3_2']);

lgraph = connectLayers(lgraph,[prefix 'Exponensial Layer uv1_1'],...
    [prefix 'Concatination Layer1 uv1/in1']); 

lgraph = connectLayers(lgraph,[prefix 'Replicate Channels Layer'],...
    [prefix 'Concatination Layer1 uv1/in2']);

lgraph = connectLayers(lgraph,[prefix 'Exponensial Layer uv2_1'],...
    [prefix 'Concatination Layer1 uv2/in1']);

lgraph = connectLayers(lgraph,[prefix 'Replicate Channels Layer'],...
    [prefix 'Concatination Layer1 uv2/in2']);

lgraph = connectLayers(lgraph,[prefix 'Exponensial Layer uv3_1'],...
    [prefix 'Concatination Layer1 uv3/in1']);

lgraph = connectLayers(lgraph,[prefix 'Replicate Channels Layer'],...
    [prefix 'Concatination Layer1 uv3/in2']);

lgraph = connectLayers(lgraph,[prefix 'Concatination Layer1 uv1'],...
    [prefix 'Hadamard Product Layer uv1']);

lgraph = connectLayers(lgraph,[prefix 'Concatination Layer1 uv2'],...
    [prefix 'Hadamard Product Layer uv2']);

lgraph = connectLayers(lgraph,[prefix 'Concatination Layer1 uv3'],...
    [prefix 'Hadamard Product Layer uv3']);

lgraph = connectLayers(lgraph,[prefix 'Hadamard Product Layer uv1'],...
    [prefix 'Concatination Layer2 uv1/in1']);

lgraph = connectLayers(lgraph,[prefix 'Exponensial Layer uv1_2'],...
    [prefix 'Concatination Layer2 uv1/in2']);

lgraph = connectLayers(lgraph,[prefix 'Hadamard Product Layer uv2'],...
    [prefix 'Concatination Layer2 uv2/in1']);

lgraph = connectLayers(lgraph,[prefix 'Exponensial Layer uv2_2'],...
    [prefix 'Concatination Layer2 uv2/in2']);

lgraph = connectLayers(lgraph,[prefix 'Hadamard Product Layer uv3'],...
    [prefix 'Concatination Layer2 uv3/in1']);

lgraph = connectLayers(lgraph,[prefix 'Exponensial Layer uv3_2'],...
    [prefix 'Concatination Layer2 uv3/in2']);

lgraph = connectLayers(lgraph,[prefix 'Concatination Layer2 uv1'],...
    [prefix 'Matrix Multiplication Layer uv1']);

lgraph = connectLayers(lgraph,[prefix 'Concatination Layer2 uv2'],...
    [prefix 'Matrix Multiplication Layer uv2']);

lgraph = connectLayers(lgraph,[prefix 'Concatination Layer2 uv3'],...
    [prefix 'Matrix Multiplication Layer uv3']);

lgraph = connectLayers(lgraph,[prefix ...
    'Matrix Multiplication Layer uv1'],[prefix 'Normalization Layer uv1']);

lgraph = connectLayers(lgraph,[prefix ...
    'Matrix Multiplication Layer uv2'],[prefix 'Normalization Layer uv2']);

lgraph = connectLayers(lgraph,[prefix ...
    'Matrix Multiplication Layer uv3'],[prefix 'Normalization Layer uv3']);

lgraph = connectLayers(lgraph,[prefix 'Normalization Layer uv1'],...
    [prefix 'Square Root Layer uv1']);

lgraph = connectLayers(lgraph,[prefix 'Normalization Layer uv2'],...
    [prefix 'Square Root Layer uv2']);

lgraph = connectLayers(lgraph,[prefix 'Normalization Layer uv3'],...
    [prefix 'Square Root Layer uv3']);

lgraph = connectLayers(lgraph,[prefix 'Square Root Layer uv1'],...
    [prefix 'Histogram Concatination Layer/in1']);

lgraph = connectLayers(lgraph,[prefix 'Square Root Layer uv2'],...
    [prefix 'Histogram Concatination Layer/in2']);

lgraph = connectLayers(lgraph,[prefix 'Square Root Layer uv3'],...
    [prefix 'Histogram Concatination Layer/in3']);

lastLayer = [prefix 'Histogram Output Layer'];
end

