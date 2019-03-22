import numpy as np
import sys
import os
import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import random
import torchvision as tv
import PIL
import random
import pickle
import sys

from dataset_processing import *
from UNet import *
from training_tools import *
from training import *


def start(args):
    # listing out hyperparameters
    hyperparameters = {}
    hyperparameters['name'] = args[0]
    hyperparameters['device'] = torch.device('cuda:'+args[1])
    hyperparameters['epochs'] = int(args[2])
    hyperparameters['batch_size'] = int(args[3])
    hyperparameters['augument_factor'] = int(args[4])
    hyperparameters['model_depth'] = int(args[5])
    hyperparameters['nonlinear_activation'] = args[6] # 'elu' or 'relu'
    hyperparameters['dropout_rate'] = float(args[7])
    hyperparameters['learning_rate'] = float(args[8])
    hyperparameters['optimizer'] = args[9] # 'adam' or 'sgd'
    hyperparameters['aug_tricks'] = args[10] #0/1/2 = none/standard/all

    masks = pickle.load(open("cache/masks.p", "rb"))
    oowl = pickle.load(open("cache/oowl.p", "rb"))

    train_keys = pickle.load(open("cache/train_keys.p","rb"))
    val_keys = pickle.load(open("cache/val_keys.p","rb"))
    test_keys = pickle.load(open("cache/test_keys.p","rb"))

    
    unet = UNet(hyperparameters)
    trainer = Trainer(unet, hyperparameters, train_keys, val_keys, oowl, masks)
    trainer.train()

if __name__ == "__main__":
    start(sys.argv[1:])
