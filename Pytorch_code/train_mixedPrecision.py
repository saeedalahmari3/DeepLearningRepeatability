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



def seed_torch(seed=2019):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    df_max = pd.DataFrame()
    task = iterationNum.split('_')[0]
    fold = iterationNum.split('_')[1]
    fold = 'Iteration_'+fold

    nb_epoch = int(nb_epoch)
    if path2Data.endswith(os.path.sep):
        path2Data = path2Data[:-1]
    expPath = os.path.dirname(path2Data)

    if not os.path.exists(os.path.join(expPath,'pytorch_exp_May2021_saveEveryEpochModel_mixedPrecision')):
        os.makedirs(os.path.join(expPath,'pytorch_exp_May2021_saveEveryEpochModel_mixedPrecision'))
    taskPath = os.path.join(expPath,'pytorch_exp_May2021_saveEveryEpochModel_mixedPrecision')
    if os.path.exists(os.path.join(taskPath,task,fold,'Models',fold+'_weights'+'.pt')) and mode == 'train':
        print('Model for iteration {} already exist, try another iteration for training'.format(iterationNum))
        sys.exit()
    mode = mode.lower()


    try:
        del model
        del best_model
    except:
        print('Seems memory already cleared from previous models')
    seed_torch(seed=int(seed))
    device = torch.device("cuda:"+GPU)
    #model = UNet(n_classes=2, padding=True, up_mode='upconv').to(device)
    model = unetmodel(in_channels=1, out_channels=2, init_features=32,BatchNorm = False)
    #print(model)
    #model.apply(weights_init)
    #model.double() # double precision
    model = weights_initFromKeras2(model)
    listOflayers,listOfWeightsMax, listOfWeightsMin = getMaximumWeights(model)
    df_max['layers'] = listOflayers
    df_max['init_max'] = listOfWeightsMax
    df_max['init_min'] = listOfWeightsMin

    #model = model.cuda()
    #print(model)
    #inputSize = (int(ch),int(img_width),int(img_hight))
    #summary(model,(int(ch),int(img_width),int(img_hight)))
    model = model.to(device)
    

    if mode == 'train':
        param_json = {}  # json for all training parameters
        param_json['seed'] = seed
        param_json['GPU'] = GPU
        param_json['total_epochs'] = nb_epoch
        param_json['batch_size'] = batch_size
        param_json['img_width'] = img_width
        param_json['img_hight'] = img_hight
        param_json['channels'] = ch
        param_json['normType'] = 'divideBY255'
        #trainImgs,trainMsks = create_train_data(trainingImages,trainingMasks,'train',img_width,img_hight)
        if not os.path.exists(os.path.join(taskPath,task,fold,'Models')):
            os.makedirs(os.path.join(taskPath,task,fold,'Models'))
        if not os.path.exists(os.path.join(taskPath,task,fold,'Summary')):
            os.makedirs(os.path.join(taskPath,task,fold,'Summary'))
        #optim = torch.optim.SGD(model.parameters(),lr=1e-4)
        optim = torch.optim.Adam(model.parameters(),lr=1e-5,eps=1e-7,amsgrad=False, weight_decay=1e-3)
        #lossFunc = nn.BCELoss(reduction='mean')
        lossFunc = torch.nn.BCEWithLogitsLoss(reduction='mean')
        lossFunc = lossFunc.to(device)
        # lr scheduler
        #exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=20, gamma=0.1)
        #listOflayers,listOfWeightsMax, listOfWeightsMin = getMaximumWeights(model)
        #df_max['layers'] = listOflayers
        #df_max['init_max'] = listOfWeightsMax
        #df_max['init_min'] = listOfWeightsMin
        df_max.to_csv(os.path.join(taskPath,task,fold,'Summary','maxPerLayer.csv'))
        '''
        total_images = len(os.listdir(os.path.join(path2Data,'folds',fold,'train','masks','train')))
        total_val_images = len(os.listdir(os.path.join(path2Data,'folds',fold,'val','masks','val')))

        print('Total Number of Images in the dataset is {}'.format(total_images))
        if os.path.exists(os.path.join(taskPath,task,fold,'Summary',fold+'_trainImgs.npy')):
            trainImgs = np.load(os.path.join(taskPath,task,fold,'Summary',fold+'_trainImgs.npy'))
        else:
            
            trainImgs,trainMasks = create_train_data(os.path.join(path2Data,'folds',fold,'train','images'),os.path.join(path2Data,'folds',fold,'train','masks'),'train',img_width,img_hight)
            np.save(os.path.join(taskPath,task,fold,'Summary',fold+'_trainImgs.npy'),trainImgs)
        mean,std = getTrainStatistics(trainImgs)
        train_data = Dataset(path2Data,os.path.join(path2Data,'folds',fold,'train','images','train'),os.path.join(path2Data,'folds',fold,'train','masks','train'),mean=mean,std=std,normalize=normType)
        val_data = Dataset(path2Data,os.path.join(path2Data,'folds',fold,'val','images','val'),os.path.join(path2Data,'folds',fold,'val','masks','val'),mean=mean,std=std,normalize=normType)
        '''
        trainingImgs = np.load(os.path.join(path2Data,'trainImgs.npy'))
        trainingMasks = np.load(os.path.join(path2Data,'trainingMasks_twoChannels.npy'))
        total_images = trainingImgs.shape[0]
        trainingImgs = trainingImgs / 255.0
        train_loader = torch.utils.data.DataLoader(trainingImgs,batch_size = batch_size, shuffle= False, num_workers = 1)
        trainMasks_loader = torch.utils.data.DataLoader(trainingMasks,batch_size = batch_size, shuffle=False,num_workers=1)
        #data_loaders = {'train': train_loader, "val": val_loader} 
        #param_json['mean'] = str(mean)
        #param_json['std'] = str(std)
        # write paramaters to json file
        with open(os.path.join(taskPath,task,fold,'Summary',fold+'_param.json'),'w') as fp:
            json.dump(str(param_json),fp)
        
        best_loss = math.inf
        # training UNet
        start_time = time.time()
        epoch_loss_dice = []
        scaler = torch.cuda.amp.GradScaler()
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

                    X = X.to(device,dtype=torch.float)  # [N, 1, H, W]
                    y = y.to(device=device,dtype=torch.float)  # [N, H, W] with class indices (0, 1)

                    optim.zero_grad()
                    with torch.cuda.amp.autocast():
                        prediction = model(X)  # [N, 2, H, W
                        prediction = F.softmax(prediction,1)
                        loss = lossFunc(prediction,y)
                        #dice = get_dice(prediction,y)
                    

                    if phase == 'train':
                        #loss.backward()
                        scaler.scale(loss).backward()
                        #optim.step()
                        scaler.step(optim)
                        scaler.update()
    
                        accum_loss += loss.item() * prediction.shape[0]
                        #accum_dice += dice.item()
                        totalImagesSeen += X.shape[0]
                        print('%d/%d \t train loss %f \r'%(totalImagesSeen,total_images,loss.item()),end="")
                    if phase == 'val':
                        accum_loss_val += loss.item() * prediction.shape[0]
                        #accum_dice_val += dice.item()
                        totalImagesSeen_val += X.shape[0]
            #print('%d/%d \t train loss %f \t val loss %f \r'%(totalImagesSeen,total_images,loss.item(),loss_val.item()),end="")
                #print('dice %f \r'%dice.item()/float(prediction.shape[0]), end="")
                
            #exp_lr_scheduler.step()

            epoch_loss = accum_loss /float(totalImagesSeen)
            #epoch_dice = accum_dice /float(totalImagesSeen)
            #epoch_loss_val = accum_loss_val / float(totalImagesSeen_val)
            #epoch_dice_val = accum_dice_val / float(totalImagesSeen_val)

            #epoch_loss_dice.append((epoch,epoch_loss,epoch_dice))

            print('epoch {}/{}, train loss {}'.format(epoch,nb_epoch,epoch_loss))
            #if (epoch > 1 and abs(epoch_loss_val - best_loss) > 1):
                #print('reducing the  weight decay by *10')
                #adjust_lr_custom(optim)
            #This code is for saving the min and max of each layer at each epoch to test why the results are not reproducible.
            #-----------------------------------------------------------------------
            listOflayers,listOfWeightsMax, listOfWeightsMin = getMaximumWeights(model)
            df_max['layers'] = listOflayers
            df_max['init_max'] = listOfWeightsMax
            df_max['init_min'] = listOfWeightsMin
            df_max.to_csv(os.path.join(taskPath,task,fold,'Summary','Epoch'+str(epoch)+'_maxPerLayer.csv'))
            epoch_model = copy.deepcopy(model)
            epoch_model = epoch_model.cpu()
            torch.save(epoch_model.state_dict(), os.path.join(taskPath,task,fold,'Models',fold+'_Epoch'+str(epoch)+'_weights.pt'))
            #------------------------------------------------------------------------
            if epoch_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_model = best_model.cpu()
                best_loss = epoch_loss
                #best_dice = epoch_dice
                try:
                    print('saving the model ...')
                    torch.save(best_model.state_dict(), os.path.join(taskPath,task,fold,'Models',fold+'_weights.pt'))
                except:
                    print('Error saving the model')
            if saveModelOption == 'Snapshot':
                if epoch%5 == 0:
                    print('saving model Snapshot...')
                    model2save = copy.deepcopy(model)
                    model2save = model2save.cpu()
                    torch.save(model2save.state_dict(), os.path.join(taskPath,task,fold,'Models',fold+'_epoch_'+str(epoch)+'_weights.pt'))
        writetoCSV(os.path.join(taskPath,task,fold,'Summary',fold+'_epoch_loss_dice'),epoch_loss_dice)
        total_time = time.time() - start_time
        print("total training time {} sec".format(total_time))
        del model
    elif mode == 'test':
        if not os.path.exists(os.path.join(taskPath,task,fold,'Models',fold+'_weights'+'.pt')):
            print('Model for {} does not exist'.format(task))
            sys.exit()
        
        #task = iterationNum.split('_')[0]
        #fold = iterationNum.split('_')[1]
        #fold = 'fold_'+fold
        for phase in ['test']:
            start_time = time.time()
            path2Model = os.path.join(taskPath,task,fold,'Models',fold+'_weights'+'.pt')
            pred_dir_test = os.path.join(taskPath,task,fold,'PredictedMasks',fold+'_'+phase+'_predMasks')
            if not os.path.exists(pred_dir_test):
                os.makedirs(pred_dir_test)
            #testImages,testIds = create_test_data(os.path.join(path2Data,'folds',fold,phase,'images'),phase,img_width,img_hight)
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
                X = X.to(device,dtype=torch.float)
                prediction = model(X)
                #prediction = F.softmax(prediction,1)
                SaveMsksToFile(prediction,ids,pred_dir_test)
            print("Total test time {}".format(time.time() - start_time))
        print('DONE')
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU', '--GPU_num', help ='GPU number (Based on nvidia-smi indexing) to use in training/testing model ', required= True)
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
