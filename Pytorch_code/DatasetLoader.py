import os
import sys
import random
import cv2
import numpy as np
import torch.utils.data as utils_data




class Dataset(utils_data.Dataset):
    def __init__(self, root_dir, image_dir, mask_dir,ch = 1, img_width = 400, img_hight = 400,normalize='center',transform=None):
        # normalize = center,divideBy255, centerandnormalize
        self.dataset_path = root_dir
        self.imgChannel = ch
        self.img_width = img_width
        self.img_hight = img_hight
        self.normalize = normalize
        #self.mean = mean
        #self.std = std
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        mask_full_path = os.path.join(self.dataset_path, self.mask_dir)
        self.mask_file_list = [f for f in os.listdir(mask_full_path) if os.path.isfile(os.path.join(mask_full_path, f))]
        random.shuffle(self.mask_file_list)
    def __len__(self):
        return len(os.listdir(os.path.join(self.dataset_path,self.mask_dir)))
    def __getitem__(self, index):
        file_name =  self.mask_file_list[index].rsplit('.', 1)[0]
        img_name = os.path.join(self.dataset_path, self.image_dir, file_name+'.png')
        mask_name = os.path.join(self.dataset_path, self.mask_dir, self.mask_file_list[index])
        image = cv2.imread(img_name,-1)  # read image
        mask = cv2.imread(mask_name,-1) # read mask
        image = cv2.resize(image,(self.img_width,self.img_hight),interpolation= cv2.INTER_CUBIC)
        mask = cv2.resize(mask,(self.img_width,self.img_hight),interpolation= cv2.INTER_LINEAR)
        if self.imgChannel == 1:
            image = np.expand_dims(image,axis=0)
        mask = (mask >= 128)*1
        image = np.array(image)
        mask = np.array(mask)
        #image = np.rollaxis(image, 2, 0)
        image = np.array(image).astype(np.float32) #float32
        mask = np.array(mask).astype(np.uint8)
        new_mask = np.empty((2,self.img_width,self.img_hight))
        Ch0 = (mask[:,:] == 0)*1
        Ch1 = (mask[:,:] == 1)*1
        new_mask[0,:,:] = Ch0
        new_mask[1,:,:] = Ch1
        new_mask = np.array(new_mask).astype(np.uint8)

        if self.normalize == 'center_normalize_batchwise':  # normalize each batch of images by substacting the mean and dividing by std
            mean = image.mean()
            std = image.std()
            image = image - mean
            image = image / std
        elif self.normalize == 'divideby255': # normalize batches of image by dividing by 255.0
            image= image = image / 255.0

        elif self.normalize == 'center': # centering the batch to have zero mean (the mean is for the whole train set)
            #image = image - self.mean
            print('Error do not use mean for pytorch 1.3')
            sys.exit()
        elif self.normalize == 'center_normalize_train': # center and normalize the batch by mean and std of the whole train set
            #image = image - self.mean
            #image = image / self.std
            print('Error do not use mean for pytorch 1.3')
            sys.exit()
        else:
            print('{} type is not recognized'.format(self.normalize))
            sys.exit()

        sample = {'image': image, 'mask': new_mask}
        if self.transform:
            sample = self.transform(sample)

        return sample
