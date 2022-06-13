
% post_process old

function [dice, postProcessedMask] = postProcessOld(mask,Disector,GT,AnnotationImage,minSize,threshold_conf)
% [path,name,ext] = fileparts(images(i).name);
% new_name = name(1:end-5);
% 
% disp(images(i).name);

% mask = imread(fullfile(pathToPredictedMsks,images(i).name));
% mask);

mask = imfill(mask,'holes');
mask = (mask < threshold_conf) ==0;
% mask);
% read Ground truth mask.
% Disector = imread(fullfile(pathToDisector,strcat(new_name,'.png')));
DisectorCropped = CropEDF_basedOnDisectorColor(Disector,Disector);
[x,y,z] = size(DisectorCropped);
%GT = imread(fullfile(pathToGT,strcat(new_name,'.png')));
GT  = imresize(GT,[x y],'nearest');
mask = imresize(mask,[x,y],'nearest');

% AnnotationImage = imread(fullfile(pathToAnnotation,strcat(new_name,'.png')));
%AnnotationImage = CropEDF_basedOnDisectorColor(AnnotationImage,AnnotationImage);
AnnotationImage = imresize(AnnotationImage,[x y],'nearest');

L = bwlabel(mask);
s = regionprops(L,'Area','PixelIdxList');
area_value = [s.Area];
% disp(area_value);
idx = find(area_value > minSize);  % threshold

bw5 = ismember(L, idx);
% bw5);

bw4_new = ApplyWatershed(bw5);

% bw4_new);
dice = dice_Coef(logical(GT),bw4_new);
L = bwlabel(bw4_new);
s = regionprops(L,'Area','PixelIdxList');
area_value = [s.Area];
% disp(area_value);
idx = find(area_value > minSize);
bw4_new = ismember(L, idx);

bw4_new = removeExclusionlineBlobs(bw4_new,'');
postProcessedMask = bw4_new;
% GT = removeExclusionlineBlobs(GT,option);
% bw4_new);
% Dice
% fprintf(Dice_fileID,'%s   \t  %d',new_name,dice);
% fprintf(Dice_fileID,'\r\n');

%imwrite(bw4_new,fullfile(pathToPost_ProcessedMsks,images(i).name));
end