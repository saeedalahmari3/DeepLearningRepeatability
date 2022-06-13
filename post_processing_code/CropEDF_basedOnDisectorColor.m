% crop baed on disector box color

function croppedImg = CropEDF_basedOnDisectorColor(EDF,countImage)
%path ='E:\NeuN.Nasiba_new_July2018\LU-24\Section1\Stack1';

%EDF = imread(fullfile(path,'EDF_and_mask','EDF_withDisector.jpeg'));
% Stack = dir(fullfile(path,'Stack','*.bmp'));
% countImage = imread(fullfile(path,'Stack',Stack(1).name));


G = countImage(:, :, 1) == 0 & countImage(:, :, 2) == 255 & countImage(:, :, 3) == 0;
R = countImage(:, :, 1) == 255 & countImage(:, :, 2) == 0 & countImage(:, :, 3) == 0;
wholeGrid = G | R;


wholeGridLines = imerode(wholeGrid, [zeros(19, 1); ones(20, 1)]) | imerode(wholeGrid, [ones(20, 1); zeros(19, 1)]) | ...
    imerode(wholeGrid, [zeros(1, 19), ones(1, 20)]) | imerode(wholeGrid, [ones(1, 20), zeros(1, 19)]);
regionWithoutExclusionLine = imfill(wholeGridLines, 'holes') & ~ R;
%imshow(regionWithoutExclusionLine);
exclusionLine = R;
inclusionLine = wholeGridLines & exclusionLine;
C = corner(regionWithoutExclusionLine);
X = C(1,1);
Y = C(1,2);
%disp(C);
Width = C(4,1) - X;
Hight = C(4,2) - Y;
rect = [X-20,Y-20,Width+40,Hight+40];
croppedImg = imcrop(EDF,rect);
%imshow(croppedImg);
end