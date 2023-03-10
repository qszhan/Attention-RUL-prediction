# -*- coding: utf-8 -*-


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from tran_esti_semi_regre import data_for_bound

assert torch.cuda.is_available(), "GPU not available"
print(torch.cuda.is_available())
import pandas as pd
from skimage import io, transform
from PIL import ImageFile
import torch
import torch.nn as nn
from model import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy
import scipy.io
import pdb

plt.ion()  # interactive mode
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_saved_model(num_fea, ds):
    model_ft = Model(num_fea)
    model_ft.cuda()
    print("=> loading checkpoint '{}'".format(os.path.join('./checkpoints', ds + '_best_RMSE.pth.tar')))
    checkpoint = torch.load(os.path.join('./checkpoints', ds + '_best_RMSE.pth.tar'))
    model_ft.load_state_dict(checkpoint['state_dict'])
    return model_ft

def define_fc_layer(model):
    return model.output

def create_data_dict(train_data, test_data):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                                 shuffle=False,
                                                 num_workers=4)
    # Create training and validation dataloaders
    dataloaders_dict = {'train': train_dataloader,
                        'val': val_dataloader}
    # dataloaders_dict = {'train': val_dataloader,
    #                     'val': train_dataloader}
    return dataloaders_dict

def forward_pass(data_loader, model, fc_layer):
    """
    a forward pass on target dataset
    :params model:
    :params fc_layer: the fc layer of the model, for registering hooks
    returns
        features: extracted features of model
        outputs: outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []

    def hook_fn_forward(module, input, pred):
        features.append(input[0].detach().cpu())
        outputs.append(pred.detach().cpu())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)
    model.eval()
    count = 0
    # for index, (inputs, labels) in enumerate(dataloader_for_feature_extraction):
    with torch.no_grad():
        for batch_index, data in enumerate(data_loader, 0):
            print('Features extracted for {} out of {} images'.format(count, len(data_loader.dataset)))
            inputs, handcrafted_feature, labels = data
            inputs, handcrafted_feature, labels = inputs.to(device), handcrafted_feature.to(device), labels.to(device)
            # model_optimizer.zero_grad()
            targets.append(labels)
            pred = model(inputs, handcrafted_feature)
            count = count + pred.shape[0]
        forward_hook.remove()
        features = torch.cat([x for x in features])
        outputs = torch.cat([x for x in outputs])
        targets = torch.cat([x for x in targets])
    return features, outputs, targets


# configs = Config()
# ===========obtain the feature using pretrained model (download from packages online)===========
print('start to load data ...')
data_name = "FD003"
# dt = "FD002"
train_dataloader , val_dataloader, num_fea = data_for_bound(data_name)
# dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
# for dt: cifar100 data
# pdb.set_trace()
print("data_load finish!")

# load model
model_ft = load_saved_model(num_fea, ds = 'FD004')
fc_layer = define_fc_layer(model_ft)
train_features, train_outputs, train_targets = forward_pass(train_dataloader, model_ft, fc_layer)
val_features, val_outputs, val_targets = forward_pass(val_dataloader, model_ft, fc_layer)
allFeatures = torch.cat((train_features, val_features), 0)
allPreds = torch.cat((train_outputs, val_outputs), 0)
allLabels = torch.cat((train_targets, val_targets), 0)

scipy.io.savemat("./features/ds_FD004/" + data_name + ".mat",
                 mdict={'features': allFeatures.to(torch.device("cpu")).numpy(),
                        'labels': allLabels.to(torch.device("cpu")).numpy(),
                        'predictions': allPreds.to(torch.device("cpu")).numpy(),
                        })
torch.save({'features': allFeatures.to(torch.device("cpu")).numpy(),
            'labels': allLabels.to(torch.device("cpu")).numpy(),
            'predictions': allPreds.to(torch.device("cpu")).numpy(),
            },
           os.path.join("./features/ds_FD004/" + data_name + ".pth"))

# # =============test the saved feature==============
# data = torch.load(os.path.join(configs.model_dir, 'resnet18-cifar10.pth'))

# pdb.set_trace()

# tar_data = torch.load(os.path.join(configs.work_dir, "features/" + 'resnet_imagenet.pth'))
# tar_probs = tar_data['probabilities']
# tar_preds = tar_data['predictions']
# x = tar_data['features']
# y = tar_data['labels']
# print(x.shape)  # array (60,000, 1000)
# print(tar_probs.shape)  # array (60,000)
