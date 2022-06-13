function [maskOfCountedCells] = getNeuronsIntersectingWithDisectorBox(predMask,DisectorCropped)
% Find inclusion/exclusion lines and the regions of interest from the
% countImage that has the disector lines to find out what regions
% should/not be counted
%imshow(DisectorCropped);
G = DisectorCropped(:, :, 1) == 0 & DisectorCropped(:, :, 2) == 255 & DisectorCropped(:, :, 3) == 0;
R = DisectorCropped(:, :, 1) == 255 & DisectorCropped(:, :, 2) == 0 & DisectorCropped(:, :, 3) == 0;
wholeGrid = G | R;
%imshow(wholeGrid);
wholeGridLines = imerode(wholeGrid, [zeros(19, 1); ones(20, 1)]) | imerode(wholeGrid, [ones(20, 1); zeros(19, 1)]) | ...
    imerode(wholeGrid, [zeros(1, 19), ones(1, 20)]) | imerode(wholeGrid, [ones(1, 20), zeros(1, 19)]);
regionWithoutExclusionLine = imfill(wholeGridLines, 'holes') & ~R;
%imshow(regionWithoutExclusionLine);
exclusionLine = R;
inclusionLine = wholeGridLines & ~exclusionLine;
% imshow(regionWithoutExclusionLine);
% imshow(predMask);
maskOfCountedCells = imreconstruct(regionWithoutExclusionLine,predMask);
% imshow(maskOfCountedCells);
end