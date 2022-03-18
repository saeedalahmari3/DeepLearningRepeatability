import torch
import math
import numpy as np
import os


# glorot_uniform init (xavier_uniform) for weights and zeros for bias

def weights_init(m): # correct

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

def weights_init2(model):
    model_newWeights = {}
    for layer, weights in model.state_dict().items():
        if layer.endswith('bias'):
            model_newWeights[layer] = torch.nn.init.zeros_(weights)
        elif layer == 'conv.weight':
            model_newWeights[layer] = torch.nn.init.xavier_uniform_(weights)
        elif layer.endswith('weight'):
            model_newWeights[layer] = torch.nn.init.xavier_uniform_(weights,gain=math.sqrt(2.0))
    model.state_dict().update(model_newWeights)
    return model

def weights_initFromKeras2(model):
    kerasWeights = np.load(os.path.join('..','kerasInitWeights','KerasInitialWeights_seed2019.npy'),allow_pickle=True)
    convCounter = 0
    model_newWeights = {}
    stat_dict = model.state_dict()
    for layer, weights in model.state_dict().items():
        layerCode = layer.split('.')
        if layer.endswith('bias'):
            stat_dict[layer].copy_(torch.nn.init.zeros_(weights))
        else:
            stat_dict[layer].copy_(torch.from_numpy(kerasWeights[convCounter]))
            convCounter = convCounter + 1
    model.state_dict().update(stat_dict)
    return model

def getMaximumWeights(model):
    listOflayers = []
    listOfMaximums = []
    listOfMinimums = []
    for layer, weights in model.state_dict().items():
        #print("Layer {}  {}".format(layer,torch.max(weights).item()))
        weightsMax = torch.max(weights).item()
        weightsMin = torch.min(weights).item()
        listOflayers.append(layer)
        listOfMaximums.append(weightsMax)
        listOfMinimums.append(weightsMin)
    return listOflayers,listOfMaximums,listOfMinimums

