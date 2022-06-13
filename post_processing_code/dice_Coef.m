%% Dice coef for two images.

function diceCoef = dice_Coef(y_true,y_pred)
%y_true = imread('C:\Users\saeed3\Google Drive\BrainImagesGrant\Attempt2018\testSet\test_masks\test\NewTestSet_Section1_Stack1.png');
%y_pred = imread('C:\Users\saeed3\Google Drive\BrainImagesGrant\Attempt2018\ASA_accepted\No_Aug\predictedTest\predictedMasks\NewTestSet_Section1_Stack1_pred.png');
smooth = 0.001;

%y_true = uint8(y_true);
[x,y] = size(y_pred);
%y_true  = imresize(y_true,[x y],'nearest');
% disp(size(y_true));
% disp(size(y_pred));


%% get Intersection.

intersection = y_true .* y_pred;

diceCoef = 2*(nnz(intersection)) / (nnz(y_true) + nnz(y_pred) + smooth);
%disp(diceCoef);
end

