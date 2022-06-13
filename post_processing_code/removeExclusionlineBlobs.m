%% Remove blobs touching the exclusion line.

%path2imgs = 'C:\Users\saeed3\Documents\MatlabCode_BrainCellCounting\maskstestPredictedNormTest2AUG2Edited';
%image = imread(fullfile(path2imgs,'NewTestSet_Section7_Stack5_pred.png'));
%images = dir(fullfile(path2imgs,'*.png'));
%for i=1: length(images)
% image = imread(fullfile(path2imgs,images(i).name));
%imshow(image);
function cells = removeExclusionlineBlobs(image,option)
%disp(size(image));
image_copy = image;
image_msk = zeros(size(image));
[x,y] = size(image);
if strcmp(option, 'widerBox_20pixels')
    image = imcrop(image,[20,20,x-40,y-40]);
end
    
wholeGrid = zeros(size(image));
[rows,cols] = size(wholeGrid);
R = wholeGrid;
G = wholeGrid;

R(:, 1:20) = 255;
R(rows-20:rows,:) = 255;
G(1,:) = 255;
G(:,cols) = 255;
wholeGrid = R | G;

wholeGridLines = imerode(wholeGrid, [zeros(19, 1); ones(20, 1)]) | imerode(wholeGrid, [ones(20, 1); zeros(19, 1)]) | ...
    imerode(wholeGrid, [zeros(1, 19), ones(1, 20)]) | imerode(wholeGrid, [ones(1, 20), zeros(1, 19)]);
regionWithoutExclusionLine = imfill(wholeGridLines, 'holes') & ~R;
%imshow(regionWithoutExclusionLine);
exclusionLine = R;
% imshow(exclusionLine);
% imshow(wholeGridLines);
inclusionLine = wholeGridLines & ~exclusionLine;
% imshow(inclusionLine);
CC = bwconncomp(image,4);
labeledImage = bwlabel(image);
cellRegions= [];
for j=1 : CC.NumObjects
    blob = ismember(labeledImage,j) > 0;
    %imshow(blob);
    cellRegions{j} = blob;
end
%imshow(image);
if (~iscell(cellRegions))
    %cellRegions = imreconstruct(regionWithoutExclusionLine, cellRegions) & ~imreconstruct(exclusionLine, cellRegions);
else
    for r = length(cellRegions): -1: 1
        if (nnz(cellRegions{r} & exclusionLine))
            cellRegions(r) = [];
        end
    end
end
if strcmp(option, 'widerBox_20pixels')
    if isempty(cellRegions)
        cells = zeros(rows,cols);
        image_msk(20:y-20,20:x-20) = cells;
        cells = image_msk;
    else
        cells = logical(sum(reshape([cellRegions{:}], size(cellRegions{1}, 1), size(cellRegions{1}, 2), []), 3));
        image_msk(20:y-20,20:x-20) = cells;
        cells = image_msk;
        end
else
    if isempty(cellRegions)
        cells = zeros(rows,cols);
    else
        cells = logical(sum(reshape([cellRegions{:}], size(cellRegions{1}, 1), size(cellRegions{1}, 2), []), 3));
    end
    
end
end