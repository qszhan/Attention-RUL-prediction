import os
import pdb

import torch
import numpy as np
from torch import optim as optim

from cal_empi_mse import select_data_semi
from model import Model
from train import Trainer
from turbofandataset import Turbofandataset

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# Detect if we have a GPU available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")





def cal_rmse_without_tl(dt):
    trainset = Turbofandataset(mode='train',
                               dataset='./datasets/CMAPSSData/train_' + dt + '_normed.txt')
    testset = Turbofandataset(mode='test',
                              dataset='./datasets/CMAPSSData/test_' + dt + '_normed.txt',
                              rul_result='./datasets/CMAPSSData/RUL_' + dt + '.txt')
    score_without_tl = []  # for empirical acc
    rmse_without_tl = []  # for empirical loss
    for rat in np.arange(0.001, 0.05, 0.001):
        # for rat in np.arange(0.01, 0.5, 0.01):
        # for rat in [0.01, 0.02, 0.04, 0.05, 0.1, 0.15]:
        """# obtain the empi_mse"""
        # reconstruct the dataset to select only [0.05, 0.1, 0.15] train and val data
        train_loader, test_loader = select_data_semi(trainset, testset, rat)
        # obtain the acc and loss if trained from scratch without TL to obtain the minus item in propsoed transferabili
        model = Model(num_fea=17)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        epochs = 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(model=model,
                          model_optimizer=optimizer,
                          print_every=50,
                          epochs=epochs,
                          device=device,
                          prefix=dt)
        best_score, best_RMSE = trainer.train(train_loader, test_loader, iteration=1)
        score_without_tl.append(best_score)
        rmse_without_tl.append(best_RMSE)
        with open('./tran_metrics_smi/rmse_without_tl/' + dt + "_1e-2.txt", 'a+',
                      encoding='utf-8') as fw:
            # fw.write(dt)
            fw.write("\n")
            fw.write("rat: %.4f\t" % rat)
            fw.write("rmse_without_TL: %.4f\t" % best_RMSE)
            fw.write("score_without_TL: %.4f\t" % best_score)
            fw.write("\n")
            fw.close()

    print("acc_without_tl, loss_without_tl", score_without_tl, rmse_without_tl)

if __name__ == "__main__":
    dt = 'FD001'
    print("dt", dt)
    cal_rmse_without_tl(dt)