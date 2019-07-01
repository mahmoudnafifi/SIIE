function lgraph = buildNet()

lrF_s = 1;
lrF_i = 0.95;
in_sz = 150; %image size
hist_sz = 61; %histogram size

% Input layer
inputLayer = imageInputLayer([in_sz in_sz 3],'Name','Input Layer',...
    'Normalization','none');

% im2double Layer
im2doubleLayer = imdoubleLayer('Im2double Layer');

layers = [inputLayer
    im2doubleLayer
    ];

%% Sensor branch

prefix_hist_sensor = 'hist_sensor ';
% add uv-RGB histogram block
[lgraph,histLayer] = add_RGB_uv_hist(layers,'Im2double Layer',...
    prefix_hist_sensor,in_sz,hist_sz);


% Conv uv-RGB Histogram (sensor) Layers
conv1_hist = convolution2dLayer(5,128,'Name',[prefix_hist_sensor 'conv1'],...
     'Stride',2,'Padding',0,'BiasLearnRateFactor',lrF_s);
int_std = sqrt(2/(5*5*3 + 128));
conv1_hist.Weights = randn([5 5 3 128]) * int_std;
conv1_hist.Bias = randn([1 1 128])*int_std;

conv2_hist = convolution2dLayer(3,256,'Name',[prefix_hist_sensor 'conv2'],...
    'Stride',2,'BiasLearnRateFactor',lrF_s);
int_std = sqrt(2/(3*3*128 + 256));
conv2_hist.Weights = randn([3 3 128 256]) * int_std;
conv2_hist.Bias = randn([1 1 256])*int_std;

conv3_hist = convolution2dLayer(2,512,'Name',[prefix_hist_sensor 'conv3'],...
    'Stride',1,'BiasLearnRateFactor',lrF_s);
int_std = sqrt(2/(2*2*256 + 512));
conv3_hist.Weights = randn([2 2 256 512]) * int_std;
conv3_hist.Bias = randn([1 1 512])*int_std;

% Relu uv-RGB Histogram (sensor) Layers
relu1_hist = reluLayer('Name',[prefix_hist_sensor 'Relu1']);
relu2_hist = reluLayer('Name',[prefix_hist_sensor 'Relu2']);
relu3_hist = reluLayer('Name',[prefix_hist_sensor 'Relu3']);


% FC uv-RGB Histogram (sensor) Layers
fc_hist = fullyConnectedLayer(9,'Name',[prefix_hist_sensor 'fc'],...
    'BiasLearnRateFactor',lrF_s);
int_std = sqrt(2/(9 + 13*13*512));
fc_hist.Weights = randn([9 13*13*512]) * int_std;

fc_hist.Bias = randn([9 1])*int_std;

%abs Layer
abs_matrix = absLayer([prefix_hist_sensor 'absLayer']);

% Matrix normalization layer
matNorm = MatrixNormalizationLayer([prefix_hist_sensor 'matrix norm'],3);


layers = [
    conv1_hist
    relu1_hist
    conv2_hist
    relu2_hist
    conv3_hist
    relu3_hist
    fc_hist
    abs_matrix
    matNorm
    ];

lgraph = addLayers(lgraph,layers);

lgraph = connectLayers(lgraph,histLayer,[prefix_hist_sensor 'conv1']);

% Additional Layers to multiply the sensor matrix by the double image

% Coefficients extraction layers
red_coffs_R_layer =  CChannelCoffExtractionLayer('Red coffs (R) layer',...
    in_sz, 9, 1,1);
red_coffs_G_layer =  CChannelCoffExtractionLayer('Red coffs (G) layer',...
    in_sz, 9, 1,2);
red_coffs_B_layer =  CChannelCoffExtractionLayer('Red coffs (B) layer',...
    in_sz, 9, 1,3);
green_coffs_R_layer =  CChannelCoffExtractionLayer(...
    'Green coffs (R) layer',in_sz, 9, 2,1);
green_coffs_G_layer =  CChannelCoffExtractionLayer(...
    'Green coffs (G) layer',in_sz, 9, 2,2);
green_coffs_B_layer =  CChannelCoffExtractionLayer(...
    'Green coffs (B) layer',in_sz, 9, 2,3);
blue_coffs_R_layer =  CChannelCoffExtractionLayer(...
    'Blue coffs (R) layer',in_sz, 9, 3,1);
blue_coffs_G_layer =  CChannelCoffExtractionLayer(...
    'Blue coffs (G) layer',in_sz, 9, 3,2);
blue_coffs_B_layer =  CChannelCoffExtractionLayer(...
    'Blue coffs (B) layer',in_sz, 9, 3,3);


lgraph = addLayers(lgraph,red_coffs_R_layer);
lgraph = addLayers(lgraph,red_coffs_G_layer);
lgraph = addLayers(lgraph,red_coffs_B_layer);

lgraph = addLayers(lgraph,green_coffs_R_layer);
lgraph = addLayers(lgraph,green_coffs_G_layer);
lgraph = addLayers(lgraph,green_coffs_B_layer);

lgraph = addLayers(lgraph,blue_coffs_R_layer);
lgraph = addLayers(lgraph,blue_coffs_G_layer);
lgraph = addLayers(lgraph,blue_coffs_B_layer);


lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Red coffs (R) layer');
lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Red coffs (G) layer');
lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Red coffs (B) layer');
lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Green coffs (R) layer');
lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Green coffs (G) layer');
lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Green coffs (B) layer');
lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Blue coffs (R) layer');
lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Blue coffs (G) layer');
lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Blue coffs (B) layer');


% inverse layer
inv_layer = InverseLayer('Sensor Matrix Inverse Layer', 9);

lgraph = addLayers (lgraph,inv_layer);

lgraph = connectLayers(lgraph,[prefix_hist_sensor 'matrix norm'],...
    'Sensor Matrix Inverse Layer');


% depth concatenation layers
catLayer_Rcoffs_R = depthConcatenationLayer(2,'Name',...
    'R + R coffs Concatination Layer');
catLayer_Rcoffs_G = depthConcatenationLayer(2,'Name',...
    'G + R coffs Concatination Layer');
catLayer_Rcoffs_B = depthConcatenationLayer(2,'Name',...
    'B + R coffs Concatination Layer');
catLayer_Gcoffs_R = depthConcatenationLayer(2,'Name',...
    'R + G coffs Concatination Layer');
catLayer_Gcoffs_G = depthConcatenationLayer(2,'Name',...
    'G + G coffs Concatination Layer');
catLayer_Gcoffs_B = depthConcatenationLayer(2,'Name',...
    'B + G coffs Concatination Layer');
catLayer_Bcoffs_R = depthConcatenationLayer(2,'Name',...
    'R + B coffs Concatination Layer');
catLayer_Bcoffs_G = depthConcatenationLayer(2,'Name',...
    'G + B coffs Concatination Layer');
catLayer_Bcoffs_B = depthConcatenationLayer(2,'Name',...
    'B + B coffs Concatination Layer');

lgraph = addLayers(lgraph,catLayer_Rcoffs_R);
lgraph = addLayers(lgraph,catLayer_Rcoffs_G);
lgraph = addLayers(lgraph,catLayer_Rcoffs_B);

lgraph = addLayers(lgraph,catLayer_Gcoffs_R);
lgraph = addLayers(lgraph,catLayer_Gcoffs_G);
lgraph = addLayers(lgraph,catLayer_Gcoffs_B);

lgraph = addLayers(lgraph,catLayer_Bcoffs_R);
lgraph = addLayers(lgraph,catLayer_Bcoffs_G);
lgraph = addLayers(lgraph,catLayer_Bcoffs_B);


lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Red Channel Splitter Layer'],...
    'R + R coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Red coffs (R) layer',...
    'R + R coffs Concatination Layer/in2');

lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Green Channel Splitter Layer'],...
    'G + R coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Red coffs (G) layer',...
    'G + R coffs Concatination Layer/in2');

lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Blue Channel Splitter Layer'],...
    'B + R coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Red coffs (B) layer',...
    'B + R coffs Concatination Layer/in2');

lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Red Channel Splitter Layer'],...
    'R + G coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Green coffs (R) layer',...
    'R + G coffs Concatination Layer/in2');

lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Green Channel Splitter Layer'],...
    'G + G coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Green coffs (G) layer',...
    'G + G coffs Concatination Layer/in2');

lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Blue Channel Splitter Layer'],...
    'B + G coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Green coffs (B) layer',...
    'B + G coffs Concatination Layer/in2');

lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Red Channel Splitter Layer'],...
    'R + B coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Blue coffs (R) layer',...
    'R + B coffs Concatination Layer/in2');

lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Green Channel Splitter Layer'],...
    'G + B coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Blue coffs (G) layer',...
    'G + B coffs Concatination Layer/in2');

lgraph = connectLayers(lgraph,...
    [prefix_hist_sensor 'Blue Channel Splitter Layer'],...
    'B + B coffs Concatination Layer/in1');
lgraph = connectLayers(lgraph,'Blue coffs (B) layer',...
    'B + B coffs Concatination Layer/in2');

%pixel-wise mult layers
hadR_R_layer = matrixHadamardProdLayer('Processed R-R',in_sz*in_sz,1);
hadR_G_layer = matrixHadamardProdLayer('Processed R-G',in_sz*in_sz,1);
hadR_B_layer = matrixHadamardProdLayer('Processed R-B',in_sz*in_sz,1);

hadG_R_layer = matrixHadamardProdLayer('Processed G-R',in_sz*in_sz,1);
hadG_G_layer = matrixHadamardProdLayer('Processed G-G',in_sz*in_sz,1);
hadG_B_layer = matrixHadamardProdLayer('Processed G-B',in_sz*in_sz,1);

hadB_R_layer = matrixHadamardProdLayer('Processed B-R',in_sz*in_sz,1);
hadB_G_layer = matrixHadamardProdLayer('Processed B-G',in_sz*in_sz,1);
hadB_B_layer = matrixHadamardProdLayer('Processed B-B',in_sz*in_sz,1);

lgraph = addLayers(lgraph,hadR_R_layer);
lgraph = addLayers(lgraph,hadR_G_layer);
lgraph = addLayers(lgraph,hadR_B_layer);

lgraph = addLayers(lgraph,hadG_R_layer);
lgraph = addLayers(lgraph,hadG_G_layer);
lgraph = addLayers(lgraph,hadG_B_layer);

lgraph = addLayers(lgraph,hadB_R_layer);
lgraph = addLayers(lgraph,hadB_G_layer);
lgraph = addLayers(lgraph,hadB_B_layer);

lgraph = connectLayers(lgraph,...
    'R + R coffs Concatination Layer','Processed R-R');
lgraph = connectLayers(lgraph,...
    'G + R coffs Concatination Layer','Processed R-G');
lgraph = connectLayers(lgraph,...
    'B + R coffs Concatination Layer','Processed R-B');

lgraph = connectLayers(lgraph,...
    'R + G coffs Concatination Layer','Processed G-R');
lgraph = connectLayers(lgraph,...
    'G + G coffs Concatination Layer','Processed G-G');
lgraph = connectLayers(lgraph,...
    'B + G coffs Concatination Layer','Processed G-B');

lgraph = connectLayers(lgraph,...
    'R + B coffs Concatination Layer','Processed B-R');
lgraph = connectLayers(lgraph,...
    'G + B coffs Concatination Layer','Processed B-G');
lgraph = connectLayers(lgraph,...
    'B + B coffs Concatination Layer','Processed B-B');

%addition layers
ProcessedR = additionLayer(3,'Name','Processed R');
ProcessedG = additionLayer(3,'Name','Processed G');
ProcessedB = additionLayer(3,'Name','Processed B');

lgraph = addLayers(lgraph,ProcessedR);
lgraph = addLayers(lgraph,ProcessedG);
lgraph = addLayers(lgraph,ProcessedB);

lgraph = connectLayers(lgraph,...
    'Processed R-R','Processed R/in1');
lgraph = connectLayers(lgraph,...
    'Processed R-G','Processed R/in2');
lgraph = connectLayers(lgraph,...
    'Processed R-B','Processed R/in3');

lgraph = connectLayers(lgraph,...
    'Processed G-R','Processed G/in1');
lgraph = connectLayers(lgraph,...
    'Processed G-G','Processed G/in2');
lgraph = connectLayers(lgraph,...
    'Processed G-B','Processed G/in3');

lgraph = connectLayers(lgraph,...
    'Processed B-R','Processed B/in1');
lgraph = connectLayers(lgraph,...
    'Processed B-G','Processed B/in2');
lgraph = connectLayers(lgraph,...
    'Processed B-B','Processed B/in3');

% Processed image output layer
catLayer_processed_image = depthConcatenationLayer(3,'Name',...
    'Processed image');

lgraph = addLayers(lgraph,catLayer_processed_image);
lgraph = connectLayers(lgraph,...
    'Processed R','Processed image/in1');
lgraph = connectLayers(lgraph,...
    'Processed G','Processed image/in2');
lgraph = connectLayers(lgraph,...
    'Processed B','Processed image/in3');



%% Add illumination branch
prefix_hist_illum = 'hist_illum ';
% first layer in the illumination branch
processedImageLayer = SkipLayer('Processed Image Output Layer'); 
% Add First uv-RGB Histogram Block
[lgraph,histLayer] = add_RGB_uv_hist(processedImageLayer,...
    'Processed Image Output Layer',prefix_hist_illum,in_sz,hist_sz,lgraph);

% Conv uv-RGB Histogram (illumination) Layers
conv1_hist = convolution2dLayer(5,128,'Name',[prefix_hist_illum 'conv1'],...
     'Stride',2,'Padding',0,'BiasLearnRateFactor',lrF_i);
int_std = sqrt(2/(5*5*3 + 128));
conv1_hist.Weights = randn([5 5 3 128]) * int_std;
conv1_hist.Bias = randn([1 1 128])*int_std;

conv2_hist = convolution2dLayer(3,256,'Name',[prefix_hist_illum 'conv2'],...
    'Stride',2,'BiasLearnRateFactor',lrF_i);
int_std = sqrt(2/(3*3*128 + 256));
conv2_hist.Weights = randn([3 3 128 256]) * int_std;
conv2_hist.Bias = randn([1 1 256])*int_std;

conv3_hist = convolution2dLayer(2,512,'Name',[prefix_hist_illum 'conv3'],...
    'Stride',1,'BiasLearnRateFactor',lrF_i);
int_std = sqrt(2/(2*2*256 + 512));
conv3_hist.Weights = randn([2 2 256 512]) * int_std;
conv3_hist.Bias = randn([1 1 512])*int_std;

% Relu uv-RGB Histogram (illumination) Layers
relu1_hist = reluLayer('Name',[prefix_hist_illum 'Relu1']);
relu2_hist = reluLayer('Name',[prefix_hist_illum 'Relu2']);
relu3_hist = reluLayer('Name',[prefix_hist_illum 'Relu3']);


% FC uv-RGB Histogram (illumination) Layers
fc_hist = fullyConnectedLayer(3,'Name',[prefix_hist_illum 'fc'],...
    'BiasLearnRateFactor',lrF_i);
int_std = sqrt(2/(3 + 13*13*512));
fc_hist.Weights = randn([3 13*13*512]) * int_std;

fc_hist.Bias = randn([3 1])*int_std;


% padding layer
paddLayer = PaddingLayer('Padded Illumination',3,9);

layers = [
    conv1_hist
    relu1_hist
    conv2_hist
    relu2_hist
    conv3_hist
    relu3_hist
    fc_hist
    paddLayer
    ];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,histLayer,[prefix_hist_illum 'conv1']);
lgraph = connectLayers(lgraph,'Processed image',...
    'Processed Image Output Layer'); 

% depth concatination layer (illumination and inv of sensor matrix)
out_data_layer = depthConcatenationLayer(2,'Name',...
    'ill and inv of sensor matrix');
lgraph  = addLayers(lgraph,out_data_layer);
lgraph = connectLayers(lgraph,'Sensor Matrix Inverse Layer',...
    'ill and inv of sensor matrix/in1'); 
lgraph = connectLayers(lgraph,'Padded Illumination',...
    'ill and inv of sensor matrix/in2'); 

% illumination layer (inv(sensor) x ill)
illLayer = IlluminantLayer('Final Illumination Layer');
lgraph = addLayers(lgraph,illLayer);
lgraph = connectLayers(lgraph,'ill and inv of sensor matrix',...
    'Final Illumination Layer');

% regression layer (angular loss)
regressionLayer = AngularLossRegression('Regression Layer (Angular Loss)');
lgraph = addLayers(lgraph,regressionLayer);
lgraph = connectLayers(lgraph,'Final Illumination Layer',...
    'Regression Layer (Angular Loss)');

%plot(lgraph);

