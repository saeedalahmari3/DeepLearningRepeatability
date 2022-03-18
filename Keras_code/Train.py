################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#   This code has been edited by SAEED ALAHMARI - University of South Floirda , Computer Science department .

#This is the correct file to train Unet. It gave me the best results.
##################################################
from Unet import *
from util import *
from tqdm import tqdm
import sys
import os
import argparse
import pandas as pd
import json
import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#taskPath = '/home/saeed3/BrainImageGrant/IterativeLearning_ICMLA2018_Orlando/IDL_redo_for_LU3_Aug2019/experiment'
def getImagesFromFile(path2Train,path2Test,SaveTo):
    trainImgs,trainMsks = create_train_data(os.path.join(path2Train,'images'),os.path.join(path2Train,'masks'),'train')
    testImgs,testIDs = create_test_data(os.path.join(path2Test,'images'),'test')
    np.save(os.path.join(SaveTo,'trainImgs.npy'),trainImgs)
    np.save(os.path.join(SaveTo,'trainMasks.npy'),trainMsks)
    np.save(os.path.join(SaveTo,'testImgs.npy'),testImgs)
    np.save(os.path.join(SaveTo,'testIds.npy'),testIDs)
    print(trainImgs.shape)
    print(trainMsks.shape)
    print(testImgs.shape)
    print(testIDs.shape)

def extractMaxFromWeights(model):
    LayerNames = []
    LayerWeights = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            weight = weights[0]
            bias = weights[1]
            LayerNames.append(layer.name+'_weights')
            LayerNames.append(layer.name+'_bias')
            LayerWeights.append(K.get_value(K.max(weight)))
            LayerWeights.append(K.get_value(K.max(bias)))
    return LayerNames,LayerWeights
def addWeightDecayL2(model,weight_decay=1e-3):
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer= regularizers.l2(weight_decay)
    return model

def trainModel(path2Data,seed,iterationNum, GPU, precision ,mode = 'train', nb_epoch = 20, batch_size = 4, img_width = 400,img_hight= 400):
    task = 'iteration_'+str(iterationNum)
    if path2Data.endswith(os.path.sep):
        path2Data = path2Data[:-1]
    expPath = os.path.dirname(path2Data)

    if not os.path.exists(os.path.join(expPath,'exp_keras_DoublePrecision')):
        os.makedirs(os.path.join(expPath,'exp_keras_DoublePrecision','Models'))
    if not os.path.exists(os.path.join(expPath,'exp_keras_DoublePrecision','Summary')):
        os.makedirs(os.path.join(expPath,'exp_keras_DoublePrecision','Summary'))
    taskPath = os.path.join(expPath,'exp_keras_DoublePrecision')
    if os.path.exists(os.path.join(taskPath,'Models',task+'_weights'+'.h5')) and mode == 'train':
        print('Model for iteration {} already exist, try another iteration name using (-iter argument) for training'.format(iterationNum))
        sys.exit()        
    
    mode = mode.lower()
    try:
        sess.close()
        K.clear_session()
    except:
        pass
    sess = init_seeds(seed=seed,precision=precision,gpu= str(GPU))
    print('Precision was set to {}'.format(precision))

    model = get_unet(1,img_width,img_hight)
    #PytorchWeightsPath = os.path.join(taskPath,'Summary','PytorchWeights_seed2019.npy')
    #Keras_setWeights(model,PytorchWeightsPath)

    print("Model Summary")
    print(model.summary())
    #plot_model(model,show_shapes=False,show_layer_names = True, dpi = 300, to_file='model_noShapes.png')
    #sys.exit()
    #df_max = pd.DataFrame()
    #LayerNames,LayerWeights = extractMaxFromWeights(model)

    #df_max['LayerName'] = LayerNames
    #df_max['Init'] = LayerWeights
    #df_max.to_csv(os.path.join(taskPath,'Summary',task+'_weightsMaxInit.csv'),index=False)

    if mode == 'train':	
        if not os.path.exists(os.path.join(taskPath,'Models')):
            os.makedirs(os.path.join(taskPath,'Models'))
        callbacks = [ModelCheckpoint(monitor='loss',
        filepath= os.path.join(taskPath,'Models',task+'_weights'+'.h5'),
                                save_best_only=True,
                                save_weights_only=False,
                                mode='min',
                                verbose=1)]
        trainingImgs = np.load(os.path.join(path2Data,'trainImgs.npy'))
        trainingMasks = np.load(os.path.join(path2Data,'trainingMasks_twoChannels.npy'))
        #trainingImgs = trainingImgs / 255.0
        
        datagen = ImageDataGenerator(rescale=1./255)

        datagen.fit(trainingImgs)
        

        #trainingMasks = ConvertMasks_faster(trainingMasks)
        np.save(os.path.join(path2Data,'trainingMasks_twoChannels.npy'),trainingMasks)
        print('Now fitting the model')
        start_time = time.time()
        #history = model.fit(x=trainingImgs, y=trainingMasks, batch_size = batch_size, epochs=nb_epoch,callbacks = callbacks, verbose=1, shuffle=False, initial_epoch=0)
        history = model.fit(datagen.flow(trainingImgs,trainingMasks,batch_size= batch_size,shuffle=False), epochs=nb_epoch, callbacks =callbacks,steps_per_epoch = len(trainingImgs)/ batch_size, verbose=1, shuffle=False, initial_epoch=0)
        print("Finished fitting the model, now saving training history...")
        total_time = time.time() - start_time
        print("Total training time {}".format(total_time))
        with open(os.path.join(taskPath,'Summary',task+'_TrainingHistory.json'), 'w') as f:
            json.dump(str(history.history), f)

    elif mode == 'test':
        start_time = time.time()
        if not os.path.exists(os.path.join(taskPath,'Models',task+'_weights'+'.h5')):
            print('Model for {} does not exist'.format(task))
            sys.exit()

        testImgs = np.load(os.path.join(path2Data,'testImgs.npy'))
        testIDs = np.load(os.path.join(path2Data,'testIds.npy'))
        testImgs = testImgs / 255.0
        model.load_weights(os.path.join(taskPath,'Models',task+'_weights'+'.h5'))

        #LayerNames,LayerWeights = extractMaxFromWeights(model)
        #df_max['LayerName'] = LayerNames
        #df_max['epoch_100'] = LayerWeights
        #df_max.to_csv(os.path.join(taskPath,'Summary',task+'_weightsMaxInit.csv'),index=False)

        # ***********  Prediction here:
        print('-'*30)
        print('Now prediction')
        imgs_mask_pred = model.predict(x = testImgs)
        #imgs_mask_pred = pred_to_imgs(imgs_mask_pred, img_width, img_hight, mode="original")
        imgs_mask_pred = imgs_mask_pred[:,1,:,:]
        
        
        pred_dir_test = os.path.join(taskPath,'PredictedMasks',task+'_predMasks')
        if not os.path.exists(pred_dir_test):
            os.makedirs(pred_dir_test)
        SaveMsksToFile(imgs_mask_pred,testIDs,pred_dir_test)
        total_time = time.time() - start_time
        print("Total test time {}".format(total_time))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU', '--GPU_num', help ='GPU number (Based on nvidia-smi indexing) to use in training/testing model ', required= True)
    parser.add_argument('-seed','--seed', help='seed value to seed numpy, tensorflow, random number generator, default=2019', default=2019)
    parser.add_argument('-d', '--dataPath', help ='path to data, make sure that names of data is correct',required= True)
    parser.add_argument('-m','--mode', help='mode: either train or test, default is train',default = 'train')
    parser.add_argument('-iter','--iteration_number', help='Iteration number of training/testing', required = True)
    parser.add_argument('-total_epochs','--total_epochs', help='Total number of epochs to run the model for, default is 20',default=20)
    parser.add_argument('-batch_size', '--batch_size', help='batch size, default is 16')
    parser.add_argument('-img_width', help='image width', default = 400)
    parser.add_argument('-img_hight', help='image hight', default = 400)
    parser.add_argument('-precision',help='precision either float32 or float64',default='float32',required=True)
    args = parser.parse_args()
 
    
    #print(args.dataPath)
    trainModel(args.dataPath,seed = int(args.seed), mode =args.mode, nb_epoch = int(args.total_epochs), iterationNum = args.iteration_number, GPU = args.GPU_num,precision=args.precision)
    #getImagesFromFile(path2Train,path2Test,path2Data)
