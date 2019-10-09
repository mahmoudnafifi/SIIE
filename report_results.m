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

clc
clear
close all;
%%% NUS
cameras = {'Canon1DsMkIII','Canon600D','FujifilmXM1',...
   'NikonD5200','OlympusEPL6','PanasonicGX1','SamsungNX2000',...
   'SonyA57'};
dataset = 'NUS';

%%% Gehler-SHI
% cameras = {'Canon1d','Canon5d'};
% dataset = 'GEHLER_SHI';


%%% CUBE+
% cameras = {'CanonEOS550D'};
% dataset = 'CUBE+';

fileName = 'results.mat';


error = [];
if isempty(cameras) == 0
    for i = 1 : length(cameras)
        load(fullfile('results',dataset,cameras{i},fileName));
        error = [error;results.angular_error];
    end
else
    load(fullfile('results',dataset,fileName));
    error = results.angular_error;
end


Mean_ae=mean(error);
Median_ae=median(error);
Best25_ae= mean(error(error<=quantile(error,0.25)));
Worst25_ae= mean(error(error>=quantile(error,0.75)));

fprintf('Results of %s:\n',dataset);
fprintf('Mean = %0.2f\n',Mean_ae);
fprintf('Median = %0.2f\n',Median_ae);
fprintf('Best 0.25 = %0.2f\n',Best25_ae);
fprintf('Worst 0.25 = %0.2f\n',Worst25_ae);
fprintf('--------------------\n');


