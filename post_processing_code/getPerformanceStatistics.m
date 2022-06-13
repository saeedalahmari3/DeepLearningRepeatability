% function for getting TP FP FN

function [Tp,Fp,Fn] = getPerformanceStatistics(predMask,countImage,fileID,name,AnnotationsMarksin)
%AnnotationsMarksin = 'blue';
% pathToAnnotation = '/home/saeed3/saeed3@mail.usf.edu/BrainImagesGrant/JournalPaper_ComputerScience_ActiveLearning/NeuN-SingleStain/Annotation_disector_countFiles/Final_Annotation';
% pathToDisector = '/home/saeed3/saeed3@mail.usf.edu/BrainImagesGrant/JournalPaper_ComputerScience_ActiveLearning/NeuN-SingleStain/Annotation_disector_countFiles/Disector_box_images';
% pathToPred = '/home/saeed3/saeed3@mail.usf.edu/BrainImagesGrant/JournalPaper_ComputerScience_ActiveLearning/NeuN-SingleStain/IterativeDeepLearning/predictedTest_folds/fold_LU2_LU3_LU14/fold_LU2_LU3_LU14_iteration1_predictedMasks_test/Postprocessed';
% 
% 
% %new_name = name(1:end-5);
% 
% %     newName = strcat('Test2_',C(2),'_',C(3));
% %     predMask = imread(fullfile(pathToMaskspred,strcat(newName{1},'_pred.png')));
% newName = name;
% predMask = imread(fullfile(pathToPred,'LU2_Section2_Stack9_pred.png'));
% countImage = imread(fullfile(pathToAnnotation,strcat('LU2_Section2_Stack9','.png')));
% %countImage = CropEDF_basedOnDisectorColor(countImage,countImage);
[x,y,z] = size(countImage);
predMask  = imresize(predMask,[x y],'nearest');

if strcmp(AnnotationsMarksin,'blue')
    B = countImage(:, :, 1) < 50 & countImage(:, :, 2) <50 & countImage(:, :, 3) >190;
    wholeGrid = B;
end

L_pred = bwlabel(predMask);
s_pred = regionprops(L_pred,'Area','PixelIdxList','Centroid','ConvexHull','ConvexImage');

L_ann = bwlabel(wholeGrid);
s_ann = regionprops(L_ann,'Area','PixelIdxList','Centroid','ConvexHull','ConvexImage');
index_ann = 1:numel(s_ann);
index_pred = 1:numel(s_pred);
TP_list = [];
for i=1 : numel(index_pred)
    if index_pred(i) == 0
        continue;
    end
    for j =1: numel(index_ann)
        if index_ann(j) == 0
            continue;
        end
        cellMask = ismember(L_pred,index_pred(i));
        %imshow(cellMask);
        CellconvexHull = bwconvhull(cellMask,'object');
        %imshow(CellconvexHull);
        
        annMask = ismember(L_ann,index_ann(j));
        %imshow(annMask);
        Intersection = CellconvexHull | annMask;
        %imshow(Intersection);
        
        cc1 = bwconncomp(Intersection,6);
        interCount  = cc1.NumObjects;
        if interCount == 1
            TP_list(end+1) = j;
            index_ann(j) = 0;
            index_pred(i) = 0;
            break;
        end
    end
end
Tp = length(TP_list);
Fp = nnz(index_pred);
Fn = nnz(index_ann);

fprintf(fileID,'%s   \t TP  %d,  \t FN %d , \t  FP  %d',name,Tp,Fn,Fp);
fprintf(fileID,'\r\n');
end