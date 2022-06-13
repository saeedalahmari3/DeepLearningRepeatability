
function [dice,processedImage] = postProcessNew(mask,Disector,GT,AnnotationImage,minSize,threshold_level)
% imshow(mask);
mask = (mask < threshold_level) ==0;
% imshow(mask);
DisectorCropped = CropEDF_basedOnDisectorColor(Disector,Disector);
[x,y,z] = size(DisectorCropped);
GT  = imresize(GT,[x y],'nearest');
mask = imresize(mask,[x,y],'nearest');

AnnotationImage = imresize(AnnotationImage,[x y],'nearest');
WorkImageA = zeros(size(mask));
AddToTheFinal = zeros(size(mask));
L = bwlabel(mask);
s = regionprops(L,'Area','ConvexArea','PixelIdxList','centroid','MajorAxisLength','MinorAxisLength','Eccentricity','EulerNumber','Solidity');
disp(s);
for i=1:numel(s)
    % Cell is Big and with Holes
    if s(i).Area <= 250
        continue;
    elseif s(i).Eccentricity < 0.9 && s(i).EulerNumber > 0 && s(i).Solidity >= 0.9
        Img = ismember(L,i);
%         imshow(Img);
        WorkImageA = or(WorkImageA,Img);
%         imshow(WorkImageA);
    elseif s(i).Solidity >= 0.9 && s(i).EulerNumber > 0
        Img = ismember(L,i);
%         imshow(Img);
        WorkImageA = or(WorkImageA,Img);
%         imshow(WorkImageA);
    elseif s(i).Solidity < 0.9 && s(i).Eccentricity >= 0.9
        Img = ismember(L,i);
%         imshow(Img);
        WorkImageA = or(WorkImageA,Img);
%         imshow(WorkImageA);
    elseif s(i).Solidity < 0.9 && s(i).MajorAxisLength/s(i).MinorAxisLength >= 1.5
        Img = imfill(ismember(L,i),'hole');
%         imshow(Img);
        WorkImageA = or(WorkImageA,Img);
%         imshow(WorkImageA);
    elseif s(i).Solidity < 0.9 && s(i).MajorAxisLength/s(i).MinorAxisLength < 1.5
        Img =ismember(L,i);
%         imshow(Img);
        AddToTheFinal = or(AddToTheFinal,Img);
%         imshow(AddToTheFinal)
    elseif s(i).EulerNumber <=0 && s(i).MajorAxisLength/s(i).MinorAxisLength >=1.5
        Img = imfill(ismember(L,i),'hole');
%         imshow(Img);
        WorkImageA = or(WorkImageA,Img);
%         imshow(WorkImageA);
    elseif s(i).EulerNumber <=0 && s(i).MajorAxisLength/s(i).MinorAxisLength < 1.5
        Img =ismember(L,i);
%         imshow(Img);
        AddToTheFinal = or(AddToTheFinal,Img);
%         imshow(AddToTheFinal)
    elseif s(i).Eccentricity >= 0.9 && s(i).MajorAxisLength/s(i).MinorAxisLength >=1.5  % longated and big
        Img = ismember(L,i);
%         imshow(Img);
        WorkImageA = or(WorkImageA,Img);
%         imshow(WorkImageA);
    else
        Img = ismember(L,i);
%         imshow(Img);
        WorkImageA = or(WorkImageA,Img);
%         imshow(WorkImageA);
    end
end

bw4_new = ApplyWatershed(WorkImageA);
% imshow(bw4_new);
bw4_new = or(bw4_new,AddToTheFinal);
% imshow(bw4_new);
dice = dice_Coef(logical(GT),bw4_new);
L = bwlabel(bw4_new);
s = regionprops(L,'Area','PixelIdxList');
area = [s.Area];
disp(area);
idx = find(area > minSize);
bw4_new = ismember(L, idx);

bw4_perim = bwperim(bw4_new);
% imshow(bw4_new);
processedImage = removeExclusionlineBlobs(bw4_new,'');
% Dice
% fprintf(Dice_fileID,'%s   \t  %d',new_name,dice);
% fprintf(Dice_fileID,'\r\n');
% mean_DICE = mean_DICE + dice;
% imwrite(bw4_new,fullfile(pathToPost_ProcessedMsks,images(i).name));
end
