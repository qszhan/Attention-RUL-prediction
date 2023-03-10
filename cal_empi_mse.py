"""
calculate the empirical mse, that is the ground-truth mse for calculate the corr
"""
import heapq
import random
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import ot
import torch
import math
from scipy.spatial.distance import cdist
import pdb
import os
from collections import Counter
from train import Trainer
from turbofandataset import Turbofandataset
from sklearn import preprocessing
from torch import optim, nn
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def select_data_semi(trainset, testset, rat):
    # Select rat% of the indices
    train_size = trainset.x.shape[0]
    train_random_indices = np.random.choice(train_size, int(train_size * rat), replace=False)
    train_x = np.array(trainset.x)
    train_y = np.array(trainset.y)
    train_mc = np.array(trainset.mean_and_coef)
    selected_train_data = train_x[train_random_indices, :]
    selected_train_target = train_y[train_random_indices]
    # pdb.set_trace()
    selected_train_mc = train_mc[train_random_indices]
    # test_size = testset.x.shape[0]
    # test_random_indices = np.random.choice(test_size, int(test_size * rat), replace=False)
    # test_x = np.array(testset.x)
    # test_y = np.array(testset.y)
    # test_mc = np.array(testset.mean_and_coef)
    # selected_test_data = test_x[test_random_indices, :]
    # selected_test_target = test_y[test_random_indices]
    # selected_test_mc = test_mc[test_random_indices]
    train_data = TensorDataset(torch.from_numpy(selected_train_data),
                               torch.from_numpy(selected_train_mc),
                               torch.from_numpy(selected_train_target))
    # test_data = TensorDataset(torch.from_numpy(selected_test_data),
    #                           torch.from_numpy(selected_test_mc),
    #                           torch.from_numpy(selected_test_target))
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4)
    return train_loader, test_loader

def initializa_model(num_fea, ds, feature_extract):
    checkpoint = torch.load(os.path.join('./checkpoints', ds + '_best_RMSE.pth.tar'))
    model_ft = Model(num_fea)
    model_ft.cuda()
    model_ft.load_state_dict(checkpoint['state_dict'])
    set_parameter_requires_grad(model_ft, feature_extract)
    # if feature_extract == true, retrain, or fine_tune
    return model_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # for param in model.parameters():
        #     param.requires_grad = False
        for param in model.lstm.parameters():
            param.requires_grad = False

        for param in model.linear.parameters():
            param.requires_grad = False

        for param in model.output.parameters():
            param.requires_grad = False


def transfer_saved_model(train_loader, test_loader, ds, dt, feature_extract, corr_indicator):
    # model_ft = initializa_model(num_fea=17)
    checkpoint = torch.load(os.path.join('./checkpoints', ds + '_best_RMSE.pth.tar'))
    model_ft = Model(num_fea=17)
    model_ft.cuda()
    model_ft.load_state_dict(checkpoint['state_dict'])
    # pdb.set_trace()
    set_parameter_requires_grad(model_ft, feature_extract)
    params_to_update = model_ft.parameters()
    optimizer = optim.Adam(params_to_update, lr=1e-2)
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model=model_ft,
                      model_optimizer=optimizer,
                      print_every=50,
                      epochs=epochs,
                      device=device,
                      prefix=dt)
    # pdb.set_trace()
    best_score, best_RMSE = trainer.train(train_loader, test_loader, 1, corr_indicator)
    return best_score, best_RMSE


def cal_empi_mse(ds, dt, feature_extract, corr_indicator):
    trainset = Turbofandataset(mode='train',
                                  dataset='./datasets/CMAPSSData/train_' + dt + '_normed.txt')
    testset = Turbofandataset(mode='test',
                              dataset='./datasets/CMAPSSData/test_' + dt + '_normed.txt',
                              rul_result='./datasets/CMAPSSData/RUL_' + dt + '.txt')
    score_TL = []  # for empirical acc
    rmse_TL = []  # for empirical loss
    for rat in np.arange(0.001, 0.05, 0.001):
    # for rat in np.arange(0.01, 0.5, 0.01):
    # for rat in [0.01, 0.02, 0.04, 0.05, 0.1, 0.15]:
        """# obtain the empi_mse"""
        # reconstruct the dataset to select only [0.05, 0.1, 0.15] train and val data
        train_loader, test_loader = select_data_semi(trainset, testset, rat)
        # fine-tune for supervised, re-train for semi-supervised
        best_score, best_RMSE = transfer_saved_model(train_loader, test_loader, ds, dt, feature_extract, corr_indicator)
        print("best_acc, best_loss", best_score, best_RMSE)
        score_TL.append(best_score)
        rmse_TL.append(best_RMSE)
        with open(os.path.join('tran_metrics_smi/' + "eval_from_TL_retrain_4_2_3_1e-2.txt"), 'a+',
                      encoding='utf-8') as fw:
            fw.write("rat: %.4f\t" % rat)
            fw.write("rmse_TL: %.4f\t" % best_RMSE)
            fw.write("score_TL: %.4f\t" % best_score)

            # fw.write("rmse_TL:\t")
            # for i in rmse_TL:
            #     fw.write(format(i, '.4f') + '\t')
            # fw.write("score_TL:\t")
            # for i in score_TL:
            #     fw.write(format(i, '.4f') + '\t')
            fw.write("\n")
            fw.close()
    print("rmse_tl", rmse_TL)


if __name__ == "__main__":
    ds = 'FD004'
    dt = 'FD002'
    print("ds, dt", ds, dt)
    # feature_extract = True: only retrain a head, else finetune whole model
    cal_empi_mse(ds, dt, feature_extract=True, corr_indicator=True)
