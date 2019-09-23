function f = calc_angular_error(source, target)
%Calculate angular error between source and target illuminants.
%
%Input:
%   -source: illuminant(s) A
%   -target: illuminant(s) B 
%
%Output:
%   -f: the angular error(s) between illuminant(s) A and illuminant(s) B.
%
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

source = source';
target = target';
target_norm = sqrt(sum(target.^2,2));
source_mapped_norm = sqrt(sum(source.^2,2));
angles=dot(source,target,2)./(source_mapped_norm.*target_norm);
angles(angles>1)=1;
f=acosd(angles);
f(isnan(f))=0;
end