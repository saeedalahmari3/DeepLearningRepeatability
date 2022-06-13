#!/usr/bin/python

################################################
#
#   Script to:
#   - Repeatability Expertiment 
#   This code has been created by SAEED ALAHMARI - Najran University and University of South Florida , Computer Science departments .
#   contact: aalahmari.saeed@gmail.com   or  https://saeedalahmari3.github.io/SaeedAlahmari/ 
##################################################


import math
import time
from util import *
import torch
import torch.utils.data
from torch.utils.data import TensorDataset
from torchsummary import summary
import torch.nn as nn
import sys
from torch.optim import lr_scheduler
import re
import random
import torch.nn.functional as F
#from unet import UNet
from unet2 import UNet as unetmodel
import copy
import argparse
import csv
from DatasetLoader import *
from init import *
import pandas as pd
import json


# Function to see all the libraries, default seed is 2019
def seed_torch(seed=2019):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Writing csv file for training history with dice and loss 
def writetoCSV(path2file,data):
    with open(path2file+'.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['epoch','train loss','train dice'])
        for row in data:
            csv_out.writerow(row)

def adjust_lr(optimizer, epoch):
    lr = 1e-4 * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def adjust_lr_custom(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 10
        param_group['weight_decay'] = param_group['weight_decay'] * 10


def showImages(mydata):
    i = 0
    sample = mydata[i]
    print(i, sample['image'].shape, sample['mask'].shape)
    cv2.imshow('image',sample['image'])
    print(sample['image'])
    print(sample['mask'])
    cv2.waitKey(0)
    cv2.imshow('mask',sample['mask'])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def trainModel(path2Data,seed,iterationNum,GPU ,mode = 'train', nb_epoch = 20, batch_size = 16, img_width = 400,img_hight= 400, ch = 1,saveModelOption='last'):

    #task = iterationNum.split('_')[0]
    fold = iterationNum
    fold = 'Iteration_'+fold

    nb_epoch = int(nb_epoch)
    if path2Data.endswith(os.path.sep):
        path2Data = path2Data[:-1]
    expPath = os.path.dirname(path2Data)

    if not os.path.exists(os.path.join(expPath,'pytorch_exp_doublePrecision')):
        os.makedirs(os.path.join(expPath,'pytorch_exp_doublePrecision'))
    taskPath = os.path.join(expPath,'pytorch_exp_doublePrecision')
    if os.path.exists(os.path.join(taskPath,fold,'Models',fold+'_weights2'+'.pt')) and mode == 'train':
        print('Model for iteration {} already exist, try another iteration for training'.format(iterationNum))
        #sys.exit()
    mode = mode.lower()


    try:
        del model
        del best_model
    except:
        print('Seems memory already cleared from previous models')
    
    seed_torch(seed=int(seed)) # Seeding 
    #device = torch.device("cuda") # Putting the model to GPU
    if GPU.lower() == 'none':
        device = torch.device("cuda")
    else:
        device = torch.device("cuda:"+GPU)


    model = unetmodel(in_channels=1, out_channels=2, init_features=32,BatchNorm = False) # Getting the model 

    model.double() # double precision
    model = weights_initFromKeras2(model) # Initalizing the model. 
    model = model.to(device)
    
    if mode == 'train':
        if not os.path.exists(os.path.join(taskPath,fold,'Models')):
            os.makedirs(os.path.join(taskPath,fold,'Models'))
        if not os.path.exists(os.path.join(taskPath,fold,'Summary')):
            os.makedirs(os.path.join(taskPath,fold,'Summary'))
        #optim = torch.optim.SGD(model.parameters(),lr=0.01)
        optim = torch.optim.Adam(model.parameters(),lr=0.001,eps=1e-7,amsgrad=False, weight_decay= 0)
        lossFunc = nn.BCELoss(reduction='mean')
        lossFunc = lossFunc.to(device)

        # Loading training data 
        trainingImgs = np.load(os.path.join(path2Data,'trainImgs.npy')) 
        trainingMasks = np.load(os.path.join(path2Data,'trainingMasks_twoChannels.npy'))
        total_images = trainingImgs.shape[0]
        trainingImgs = trainingImgs / 255.0
        train_loader = torch.utils.data.DataLoader(trainingImgs,batch_size = batch_size, shuffle= False, num_workers = 1)
        trainMasks_loader = torch.utils.data.DataLoader(trainingMasks,batch_size = batch_size, shuffle=False,num_workers=1)
       
        best_loss = math.inf
        # training UNet
        start_time = time.time()
        epoch_loss_dice = []
        print('Training started ...')
        for epoch in range(1,nb_epoch+1):
            accum_loss = 0.0
            accum_dice = 0.0
            accum_loss_val = 0.0
            accum_dice_val = 0.0

            totalImagesSeen = 0
            totalImagesSeen_val = 0
            for phase in ['train']:

                for X,y in zip(train_loader,trainMasks_loader):
                    #X, y = batch['image'], batch['mask']

                    X = X.to(device,dtype=torch.double)  # [N, 1, H, W]
                    y = y.to(device=device,dtype=torch.double)  # [N, H, W] with class indices (0, 1)

                    optim.zero_grad()
                    prediction = model(X)  # [N, 2, H, W
                    prediction = F.softmax(prediction,1)
                    loss = lossFunc(prediction,y)
                    dice = get_dice(prediction,y)

                    if phase == 'train':
                        loss.backward()
                        optim.step()
    
                        accum_loss += loss.item() * prediction.shape[0]
                        accum_dice += dice.item()
                        totalImagesSeen += X.shape[0]
                        print('%d/%d \t train loss %f \r'%(totalImagesSeen,total_images,loss.item()),end="")
                    if phase == 'val':
                        accum_loss_val += loss.item() * prediction.shape[0]
                        accum_dice_val += dice.item()
                        totalImagesSeen_val += X.shape[0]

            epoch_loss = accum_loss /float(totalImagesSeen)
            epoch_dice = accum_dice /float(totalImagesSeen)

            epoch_loss_dice.append((epoch,epoch_loss,epoch_dice))

            print('epoch {}/{}, train loss {}, train dice {}'.format(epoch,nb_epoch,epoch_loss,epoch_dice))

            #------------------------------------------------------------------------
            if epoch_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_model = best_model.cpu()
                best_loss = epoch_loss
                best_dice = epoch_dice
                try:
                    print('saving the model ...')
                    torch.save(best_model.state_dict(), os.path.join(taskPath,fold,'Models',fold+'_weights2.pt'))
                except:
                    print('Error saving the model')
            if saveModelOption == 'Snapshot':
                if epoch%5 == 0:
                    print('saving model Snapshot...')
                    model2save = copy.deepcopy(model)
                    model2save = model2save.cpu()
                    torch.save(model2save.state_dict(), os.path.join(taskPath,fold,'Models',fold+'_epoch_'+str(epoch)+'_weights.pt'))
        writetoCSV(os.path.join(taskPath,fold,'Summary',fold+'_epoch_loss_dice'),epoch_loss_dice)
        total_time = time.time() - start_time
        print("total training time {} sec".format(total_time))
        f = open(os.path.join(taskPath,fold,'Summary','training_time.txt'),'w')
        f.write('training time is \n')
        f.write(total_time)
        f.write('sec')
        f.close()
        del model
        print('Done')

        
    elif mode == 'test':
        if not os.path.exists(os.path.join(taskPath,fold,'Models',fold+'_weights2'+'.pt')):
            print('Model for does not exist')
            sys.exit()

        for phase in ['test']:
            start_time = time.time()
            path2Model = os.path.join(taskPath,fold,'Models',fold+'_weights2'+'.pt')
            pred_dir_test = os.path.join(taskPath,fold,'PredictedMasks2','predMasks')
            if not os.path.exists(pred_dir_test):
                os.makedirs(pred_dir_test)
            # Loading test data 
            testImages = np.load(os.path.join(path2Data,'testImgs.npy'))
            testIds = np.load(os.path.join(path2Data,'testIds.npy'))
            testImages = testImages / 255.0
            print('Total number of test Images is {}'.format(testImages.shape[0]))
            dataloader = torch.utils.data.DataLoader(testImages, batch_size=1,
                                                shuffle=False)
        
            model.load_state_dict(torch.load(path2Model))
            model.eval()
            print('Now predicting on '+phase+' set, please wait...')
            for X,ids in tqdm(zip(dataloader,testIds)):
                X = X.to(device,dtype=torch.double)
                prediction = model(X)
                prediction = F.softmax(prediction,1)
                SaveMsksToFile(prediction,ids,pred_dir_test)
            total_time = time.time() - start_time
            print("Total test time {}".format(total_time))
            f = open(os.path.join(taskPath,fold,'Summary','testing_time.txt'),'w')
            f.write('testing time is \n')
            f.write(total_time)
            f.write('sec')
            f.close()
        print('DONE')
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU', '--GPU_num', help ='GPU number (Based on nvidia-smi indexing) to use in training/testing model ', required= False,default=None)
    parser.add_argument('-seed','--seed', help='seed value to seed numpy, tensorflow, random number generator', default=  2019)
    parser.add_argument('-d', '--dataPath', help ='path to data, make sure that names of data is correct',required= True)
    parser.add_argument('-m','--mode', help='mode: either train or test, default is train',default = 'train')
    parser.add_argument('-iter','--iteration_number', help='Iteration number of training/testing', required = True)
    parser.add_argument('-total_epochs','--total_epochs', help='Total number of epochs to run the model for, default is 20',default=20)
    parser.add_argument('-batch_size', '--batch_size', help='batch size, default is 16',default = 16)
    parser.add_argument('-img_width', help='image width', default = 400)
    parser.add_argument('-img_hight', help='image hight', default = 400)
    parser.add_argument('-ch','--channel', help='Number of channels of input data, gray images channels =1, RGB images channels = 3, default=1',default=1)
    parser.add_argument('-saveOption','--saveOption',help='Save model every certain number of epochs or the last model', default='last')
    parser.add_argument('-normType','--normType', help='Normalization type name, can be <center>,<center_normalize_batchwise>,<divideby255>,<center_normalize_train>, default <center>',default = 'center')
    args = parser.parse_args()
 
    #print(args.dataPath)
    trainModel(args.dataPath,seed= int(args.seed),ch=args.channel,nb_epoch = args.total_epochs,batch_size = int(args.batch_size), mode =args.mode, iterationNum = args.iteration_number, GPU = args.GPU_num,saveModelOption = args.saveOption)
