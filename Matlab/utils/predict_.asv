function est_ill = predict_(model, image, device, save_mapped)
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

if nargin == 3
    save_mapped = 1;
end

image = im2double(image);

params = {'hist_sensor_conv1_Weights', 'hist_sensor_conv1_Bias',...
    'hist_sensor_conv2_Weights', 'hist_sensor_conv2_Bias', ...
    'hist_sensor_conv3_Weights', 'hist_sensor_conv3_Bias', ...
    'hist_sensor_fc_Weights', 'hist_sensor_fc_Bias', ...
    'hist_illum_conv1_Weights', 'hist_illum_conv1_Bias', ...
    'hist_illum_conv2_Weights', 'hist_illum_conv2_Bias', ...
    'hist_illum_conv3_Weights', 'hist_illum_conv3_Bias', ...
    'hist_illum_fc_Weights', 'hist_illum_fc_Bias'};

hist_sensor = RGBuvHistBlock('Hist_Block_sensor',150,61,...
    [model.hist_sensor_C1, model.hist_sensor_C2, model.hist_sensor_C3],...
    [model.hist_sensor_sigma_uv1_1, model.hist_sensor_sigma_uv2_1,...
    model.hist_sensor_sigma_uv3_1],...
    [model.hist_sensor_sigma_uv1_2, model.hist_sensor_sigma_uv2_2,...
    model.hist_sensor_sigma_uv3_2],1);

hist_ill = RGBuvHistBlock('Hist_Block_illuminant',150,61,...
    [model.hist_illum_C1, model.hist_illum_C2, model.hist_illum_C3],...
    [model.hist_illum_sigma_1_1, model.hist_illum_sigma_2_1,...
    model.hist_illum_sigma_3_1],...
    [model.hist_illum_sigma_1_2, model.hist_illum_sigma_2_2,...
    model.hist_illum_sigma_3_2],1);

sz = size(image);

if strcmpi(device,'cpu')
    for i = 1 : length(params)
        eval(sprintf('model.%s = dlarray(model.%s);', params{i}, params{i}));
    end
    image = dlarray(image,'SSC');
else
    for i = 1 : length(params)
        eval(sprintf('model.%s = dlarray(gpuArray(model.%s));', ...
            params{i}, params{i}));
    end
    image = dlarray(gpuArray(image),'SSC');
end

out = dlarray(hist_sensor.predict(image),'SSC');

for i = 1 : 3
    if i == 3
        stride = 1;
    else
        stride = 2;
    end
    eval(sprintf(['out = dlconv(out,model.hist_sensor_conv%d_Weights,' ...
        'model.hist_sensor_conv%d_Bias,''Padding'',0,''Stride'',%d);'],...
        i, i,stride));
    out = relu(out);
end
sensor_matrix = reshape(abs(fullyconnect(out,model.hist_sensor_fc_Weights, ...
    model.hist_sensor_fc_Bias)),[3,3]);

n = norm(extractdata(sensor_matrix),1) + 0.0001;
sensor_matrix = sensor_matrix ./ n;

out = reshape(reshape(image,[],3) * sensor_matrix',...
    [sz(1) sz(2) sz(3)]);

if save_mapped == 1
    if strcmpi(device,'cpu')
        imwrite(extractdata(out),'mapped.png');
    else
        imwrite(gather(extractdata(out)),'mapped.png');
    end
end

out = dlarray(hist_ill.predict(out),'SSC');

for i = 1 : 3
    if i == 3
        stride = 1;
    else
        stride = 2;
    end
    eval(sprintf(['out = dlconv(out,model.hist_illum_conv%d_Weights,' ...
        'model.hist_illum_conv%d_Bias,''Padding'',0,''Stride'',%d);'],...
        i,i,stride));
    out = relu(out);
end
sensor_matrix = extractdata(sensor_matrix);
if det(sensor_matrix) == 0
    sensor_matrix = sensor_matrix + rand(3,3)/1000;
end
est_ill = pinv(sensor_matrix) * ...
    extractdata(fullyconnect(out,model.hist_illum_fc_Weights, ...
    model.hist_illum_fc_Bias));


if strcmpi(device,'gpu')
    est_ill = gather(est_ill);
end
