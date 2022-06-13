% This code is to post-process and count neurons in Predicted masks using
% Unet.
%check threshold of small objects. IMPORTANT
%%%% Requirments.
% Path to predicted masks.
% Path to cropped (annotation) around disector line for visualization.
% confusion matrix and count will be save in files called count.txt and
% confusionMatrix.txt

Path2Experiment = '../test_exp';  % Path to experiment example (pytorch_doublePrecision)
pathToAnnotation = '../data/Annotation_disector_countFiles/Final_Annotation';
pathToDisector = '../data/Annotation_disector_countFiles/Disector_box_images';
pathToGT = '../data/Annotation_disector_countFiles/masks';
iterations = {'Iteration_1','Iteration_2','Iteration_3','Iteration_4','Iteration_5','Iteration_6','Iteration_7'};

for f =1 : numel(iterations)
    folderName = iterations{f};
    print(folderName);
        postProcessingType = 'OldPostprocessing_0.5'; threshold_level = 128;
        Folds_iterate = 0; % Is this is folds based or just testing on a single mouse only once.
        minSize = 250;

        TaskPath = fullfile(Path2Experiment,folderName,'PredictedMasks2','predMasks');
        pathToPredictedMsks = TaskPath;
        
        if(~exist(fullfile(TaskPath,postProcessingType),'dir'))
            mkdir(fullfile(TaskPath,postProcessingType));
        end
        
        if(~exist(fullfile(TaskPath,postProcessingType,'Postprocessed'),'dir'))
            mkdir(fullfile(TaskPath,postProcessingType,'Postprocessed'));
            mkdir(fullfile(TaskPath,postProcessingType,'Visualized'));
        end
        pathToPost_ProcessedMsks = fullfile(TaskPath,postProcessingType,'Postprocessed');

        
        %%  STEP 1)       Post-Processing.
        %disp(fold(z).name(end-5:end));
        Dice_fileID = fopen(fullfile(TaskPath,postProcessingType,strcat(folderName,'_diceCoef.txt')),'w');
        count_fileID = fopen(fullfile(TaskPath,postProcessingType,strcat(folderName,'_count.txt')),'w');
        saveTopath = fullfile(TaskPath,postProcessingType,'Visualized');
        CM_fileID = fopen(fullfile(TaskPath,postProcessingType,strcat(folderName,'_confusionMatrix.txt')),'w');
        images = dir(fullfile(pathToPredictedMsks,'*.png'));
        mean_DICE = 0.0;
        Total_FN = 0;
        Total_FP = 0;
        Total_TP=0;
        for i=1: length(images)
            if startsWith(images(i).name,'.')
                continue
            end
            %% load data
            [path,name,ext] = fileparts(images(i).name);
            disp(name);
            new_name = name(1:end-5);
            mask = imread(fullfile(pathToPredictedMsks,images(i).name));
            Disector = imread(fullfile(pathToDisector,strcat(new_name,'.png')));
            AnnotationImage = imread(fullfile(pathToAnnotation,strcat(new_name,'.png')));
            GT = imread(fullfile(pathToGT,strcat(new_name,'.png')));
            if strcmp(postProcessingType,'UpdatedPostprocessing_0.5')
                [dice,postProcessedMask] = postProcessNew(mask,Disector,GT,AnnotationImage,minSize,threshold_level);
            elseif strcmp(postProcessingType,strcat('OldPostprocessing_0.5'))
                [dice,postProcessedMask] = postProcessOld(mask,Disector,GT,AnnotationImage,minSize,threshold_level);
            else
                error('postProcessType is not defined');
            end
            fprintf(Dice_fileID,'%s   \t  %d',new_name,dice);
            fprintf(Dice_fileID,'\r\n');
            mean_DICE = mean_DICE + dice;
            imwrite(postProcessedMask,fullfile(pathToPost_ProcessedMsks,images(i).name));
            % get count and visualization
            [vis,MaskForStatistics,totalNumOfCells] = getCount(postProcessedMask,AnnotationImage,Disector);
            fprintf(count_fileID,'%s   \t  %d',name,totalNumOfCells);
            fprintf(count_fileID,'\r\n');
            imwrite(vis,fullfile(saveTopath,strcat(new_name,'.png')));
            % get TP FP FN
            [Tp,Fp,Fn] = getPerformanceStatistics(MaskForStatistics,AnnotationImage,CM_fileID,name,'blue');
            Total_TP = Total_TP + Tp;
            Total_FP = Total_FP + Fp;
            Total_FN = Total_FN + Fn;
        end
        fclose(Dice_fileID);
        fclose(CM_fileID);
        fclose(count_fileID);
end
