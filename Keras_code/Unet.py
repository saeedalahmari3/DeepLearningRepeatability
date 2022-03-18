# Unet

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau,EarlyStopping
#from keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys
import os
import math
import keras
import random as rn
import shutil
import cv2
from tensorflow.keras import backend as K
print("Keras version is {}".format(keras.__version__))
print('Deprecated warnings are disabled')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def init_seeds(gpu,precision,seed=2019):
    np.random.seed(seed)
    rn.seed(seed)
    session_conf = tf.compat.v1.ConfigProto()
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
    #session_conf.gpu_options.visible_device_list = gpu
    os.environ['TF_CUDNN_DETERMINISTIC'] ='true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    from tensorflow.keras import backend as K
    K.set_image_data_format('channels_first')
    tf.random.set_seed(seed)
    tf.keras.backend.set_floatx(precision) # setting the precision
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    return sess

#def dice_coef(y_true, y_pred):
#    smooth = 1.
#    y_true = y_true[:,1,:,:]
#    y_pred = y_pred[:,1,:,:]
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true,y_pred,batch_size=4):
    smooth = 0.001
    dice_accum = 0.0
    total_masks = y_true.shape.as_list()
    dice_mean = 0.0
    if (batch_size != total_masks[0] and total_masks[0] != None):
        batch_size = total_masks[0]
    for i in range(0,batch_size):
        y = y_true[i,1,:,:]
        p = y_pred[i,1,:,:]
        intersection = K.sum(y*p)
        dice = (2. * intersection) / (K.sum(y) + K.sum(p) + smooth)
        dice_accum = dice_accum + dice
    dice_mean = dice_accum / float(batch_size)
    return dice_mean

def Keras_setWeights(model,path2Weights):
    PytorchWeights = np.load(path2Weights,allow_pickle=True)
    print(PytorchWeights.shape)
    counter = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            weight = weights
            weight[0] = PytorchWeights[counter]
            #print(weight[0].shape)
            counter = counter + 1
            weight[1] = PytorchWeights[counter]
            #print(weight[1].shape)
            counter = counter + 1
            layer.set_weights(weight)
            #print('max {}  min {}'.format(np.amax(weight[0]),np.amin(weight[0])))
    return

def get_unet(ch,img_rows, img_cols):
    inputs = Input((ch,img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='relu',padding='same')(conv9)
    #conv10 = core.Reshape((2,img_rows*img_cols))(conv10)
    #conv10 = core.Permute((2,1))(conv10)
    #conv11 = core.Activation('softmax')(conv10)
    conv11 = keras.layers.Softmax(axis=1)(conv10)
    model = Model(inputs=[inputs], outputs=[conv11])

    model.compile(optimizer=Adam(lr=1e-3,amsgrad=False),loss='binary_crossentropy', metrics=[dice_coef])

    return model
