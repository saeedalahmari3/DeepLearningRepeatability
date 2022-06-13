
%% 
 function newMask = ApplyWatershed(bw)
    bw2 = bw;
    D = -bwdist(~bw2);
    Ld = watershed(D);
    bw2 = bw;
    bw2(Ld == 0) = 0;
    mask = imextendedmin(D,2);
    %imshow(mask);
    D2 = imimposemin(D,mask);
    Ld2 = watershed(D2);
    bw3 = bw;
    bw3(Ld2 == 0) = 0;
    newMask = bw3;
    %imshow(newMask);
  end