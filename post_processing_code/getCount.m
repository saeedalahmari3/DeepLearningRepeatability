% This function is to compare ground truth with the segmented image.

function [vis,predMask,totalNumberOfCells] = getCount(predMask,Annotation,Disector)

        DisectorCropped = CropEDF_basedOnDisectorColor(Disector,Disector);
        [x,y,z] = size(DisectorCropped);
        
        Annotation = imresize(Annotation,[x,y]);
        
        predMask = logical(predMask);
        
        [x,y,z] = size(DisectorCropped);
        % Get neurons that intersection with disector line.
        predMask = getNeuronsIntersectingWithDisectorBox(predMask,DisectorCropped);
        cc2 = bwconncomp(predMask,8);
        totalNumberOfCells  = cc2.NumObjects;
        predMask  = imresize(predMask,[x y],'nearest');
        Annotation = imresize(Annotation,[x,y],'nearest');
        task1_perim = bwperim(predMask);
        vis = imoverlay(Annotation,task1_perim,'red');
        
        %imwrite(vis,fullfile(saveTopath,files(i).name));
%         fprintf(fileID,'%s   \t  %d',name,number2);
%         fprintf(fileID,'\r\n');
end

