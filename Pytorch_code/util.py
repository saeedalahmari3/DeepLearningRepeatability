# util file
import numpy as np
import os
import torch
import cv2
from tqdm import tqdm
import torch.nn as nn
import sys


def saveImageANDMask(image,mask,Path2File):
    cv2.imwrite(path2File+'.png',image)
    cv2.imwrite(path2File+'mask.png',mask)

def getTrainStatistics(trainImgs):
    mean = trainImgs.mean()
    std = trainImgs.std()
    return mean,std
def dice_coef_np(y_true,y_pred):
    smooth = 1.0
    y_true = y_true
    y_pred = y_pred[:,1,:,:]
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def get_dice(pred, target, smooth = 0.001):
    target = target.float()
    target = target[:,1,:,:]
    pred = pred[:,1,:,:]
    pred = pred.contiguous()
    target = target.contiguous()
    dice_sum = 0.0
    for i in range(target.size()[0]):
        intersection = torch.sum(pred[i,:,:] * target[i,:,:])
        dice = (2. * intersection) / (torch.sum(pred[i,:,:]) + torch.sum(target[i,:,:]) + smooth)
        dice_sum = dice_sum + dice
    return dice_sum

def create_train_data(data_path,mask_path,setName,img_width, img_hight):  # SetName string = 'train' or 'val'
    train_data_path = os.path.join(data_path, setName)
    train_mask_path = os.path.join(mask_path, setName)
    images = os.listdir(train_data_path)
    total = len(images) / 2
    imgs = np.ndarray((len(images),1, img_width, img_hight),dtype=np.float32)#dtype='float64'    dtype=np.uint8 resize() got an unexpected keyword argument 'preserve_range'
    imgs_mask = np.ndarray((len(images),1,img_width, img_hight), dtype=np.uint8)  #dtype=np.uint8

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in tqdm(images):
        image_mask_name = image_name
        img = cv2.imread(os.path.join(train_data_path, image_name),-1)
        img_mask = cv2.imread(os.path.join(train_mask_path, image_mask_name),-1)
        img = cv2.resize(img,(img_width,img_hight),interpolation= cv2.INTER_CUBIC)
        img_mask = cv2.resize(img_mask,(img_width,img_hight),interpolation= cv2.INTER_LINEAR)
        img_mask = (img_mask >= 128)*1    # Sep 17 2018, removed *255
        img = np.array([img])
        img_mask = np.array([img_mask])
        #print(img_mask)
        imgs[i] = img
        imgs_mask[i] = img_mask
        i += 1
    print('Loading done.')
    return imgs,imgs_mask

def create_test_data(data_path,setName,img_width, img_hight):
    train_data_path = os.path.join(data_path,setName)
    images = os.listdir(train_data_path)
    total = len(images)
    imgs = np.ndarray((total, 1 ,img_width, img_hight),dtype=np.float32)
    imgs_id = np.ndarray((total, ),dtype='a30')   #dtype='a30' dtype=np.int32
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in tqdm(images):
        img_id = image_name.split('.')[0]
        img = cv2.imread(os.path.join(train_data_path, image_name),-1)
        img = cv2.resize(img,(img_width,img_hight),interpolation= cv2.INTER_CUBIC)
        img = np.array([img])
        imgs[i] = img
        imgs_id[i] = img_id
        i += 1
    print('Loading done.')
    return imgs, imgs_id

def ConvertMasks(masks):
    print("Mask shape before {}".format(masks.shape))
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    #masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        Ch0 = (masks[i,0,:,:] == 0)*1
        Ch1 = (masks[i,0,:,:] == 1)*1
        new_masks[i,:,0] = Ch0.flatten()
        new_masks[i,:,1] = Ch1.flatten()
    #print("Mask shape after {}".format(new_masks.shape))
    return new_masks

def GetTwoChannelMask(masks):
    assert(len(masks.shape)==4)
    assert(masks.shape[1]==1)
    print('-'*30)
    print('Creating two channel masks')
    print('-'*30)
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    new_masks = np.empty((masks.shape[0],2,im_h,im_w))
    print(new_masks.shape)
    for i in tqdm(range(masks.shape[0])):
        Ch0 = (masks[i,0,:,:] == 0)*1
        Ch1 = (masks[i,0,:,:] == 1)*1
        new_masks[i,0,:,:] = Ch0
        new_masks[i,1,:,:] = Ch1
    return new_masks

'''
def GetTwoChannelMask(mask):
    assert(len(mask.shape)==3)
    assert(mask.shape[0]==1)
    im_h = mask.shape[1]
    im_w = mask.shape[2]
    new_mask = np.empty((2,im_h,im_w))
    Ch0 = (mask[0,:,:] == 0)*1
    Ch1 = (mask[0,:,:] == 1)*1
    new_mask[0,:,:] = Ch0
    new_mask[1,:,:] = Ch1
    return new_mask
'''

def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images


def SaveMsksToFile(imgs_mask_pred,testIDs,saveToFolder):
    #print(testIDs)
    try:
        imgs_mask_pred = imgs_mask_pred.cpu()
        imgs_mask_pred = imgs_mask_pred.detach().numpy()
    except:
        print('predictions are not torch tensor, will proceed now ...')
    if not os.path.exists(saveToFolder):
       os.mkdir(saveToFolder)
    if imgs_mask_pred.shape[0] > 1:
        for image,testId in zip(imgs_mask_pred,testIDs):
            image = (image[1,:,:]*255).astype(np.uint8)
            cv2.imwrite(os.path.join(saveToFolder, testId.decode("utf-8") + '_pred.png'), image)
    else:
        for image in imgs_mask_pred:
            image = (image[1,:,:]*255).astype(np.uint8)
            cv2.imwrite(os.path.join(saveToFolder, testIDs.decode("utf-8") + '_pred.png'), image)
    return 
