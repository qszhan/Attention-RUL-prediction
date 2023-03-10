import heapq
import random
import time
from torch.utils.data import DataLoader
import numpy as np
import ot
import torch
import math
from scipy.spatial.distance import cdist
import pdb
import os
from collections import Counter

from compute_corr import calculate_cor
from turbofandataset import Turbofandataset

# assert torch.cuda.is_available(), "GPU not available"
# print(torch.cuda.is_available())
from sklearn import preprocessing
from torch import optim, nn
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms

# import geomloss


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# Detect if we have a GPU available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def bound_estimation_semi(src_err, src_x, src_y, tar_x, tar_y, lipschitz_constant):
    starttime = time.time()
    pdb.set_trace()
    # Computes the squared Euclidean distance between vectors
    fcost_x = cdist(src_x, tar_x, metric='sqeuclidean') # (17659, 21)
    alpha = 1 / fcost_x.max()
    # alpha = 0.001
    Px, log = ot.emd(ot.unif(src_x.shape[0]), ot.unif(tar_x.shape[0]), fcost_x, log=True)
    print("log.cost", log['cost'])
    my_wx = np.sum(Px * fcost_x) ** 0.5
    # wassserstein_disx = ot.sinkhorn2(ot.unif(src_x.shape[0]), ot.unif(tar_x.shape[0]), fcost_x, reg=0.1)
    wassserstein_disx_emd2 = ot.emd2(ot.unif(src_x.shape[0]), ot.unif(tar_x.shape[0]), fcost_x)
    num_labeled = len(tar_y)
    Es = np.mean(src_y[num_labeled:])
    # src_y, tar_y should have the same number of columns
    # fcost_y = cdist(src_y[:num_labeled], tar_y[:num_labeled], metric='sqeuclidean')
    src_y = np.array(src_y[:num_labeled]).reshape(-1, 1)
    tar_y = np.array(tar_y[:num_labeled]).reshape(-1, 1)
    fcost_y = cdist(src_y, tar_y, metric='sqeuclidean')
    Py = ot.emd(ot.unif(src_y.shape[0]), ot.unif(tar_y.shape[0]), fcost_y)
    # wassserstein_disy = ot.sinkhorn2(ot.unif(src_y[:num_labeled].shape[0]), ot.unif(tar_y[:num_labeled].shape[0]),fcost_y, reg=0.1)

    wassserstein_disy_emd2 = ot.emd2(ot.unif(src_y.shape[0]), ot.unif(tar_y.shape[0]), fcost_y)
    my_wy = np.sum(Py * fcost_y) ** 0.5
    # alpha = lipz*lamda constant = lipz*f(lamda)
    lamda = alpha / lipschitz_constant
    constant = 2 * lipschitz_constant * np.exp(-lamda)
    print("constant", constant)
    # taking f as e(-x)
    # transferability_ot2 = src_err + 2 * alpha * wassserstein_disx_emd2 + wassserstein_disy_emd2 + Es + lipschitz_constant - loss_without_tl
    # transferability_sinkhorn2 = src_err + 2 * alpha * wassserstein_disx + wassserstein_disy  + Es + lipschitz_constant - loss_without_tl
    # lipz_c_transferability = src_err + 2 * alpha * wassserstein_disx_emd2 + wassserstein_disy_emd2 + Es + lipschitz_constant - loss_without_tl
    c_my_bound = src_err + 2 * alpha * my_wx + my_wy + Es + constant
    c_ot2_bound = src_err + 2 * alpha * wassserstein_disx_emd2 + wassserstein_disy_emd2 + Es + constant

    # pdb.set_trace()
    print("alpha", alpha)
    print("Es", Es)
    print("wdisx_emd2, wdisy_emd2, c_ot2_bound", wassserstein_disx_emd2, wassserstein_disy_emd2, c_ot2_bound)
    print("my_wx, my_wy, c_my_bound", my_wx, my_wy, c_my_bound)
    endtime = time.time()
    return alpha, c_ot2_bound, c_my_bound, Es, (endtime - starttime)


def main_with_N_K(ds, dt, mse_ds):
    # step 1: load the saved features and labels for the Ds and Dt"""
    sre_data = torch.load(os.path.join('./features/ds_' + ds, ds + ".pth"))
    src_features = sre_data['features']
    src_labels = sre_data['labels']
    tar_data = torch.load(os.path.join('./features/ds_' + ds, dt + ".pth"))
    tar_features = tar_data['features']
    tar_labels = tar_data['labels']
    ot2_bound_all = []
    my_bound_all = []
    alpha = []
    mse = []
    loss = []
    for rat in np.arange(0.001, 0.05, 0.001):
    # for rat in np.arange(0.01, 0.5, 0.01):
    # for rat in [0.01, 0.02, 0.04, 0.05, 0.1, 0.15]:
        # obtain the acc and loss if trained from scratch without TL to obtain the minus item in propsoed transferabili
        # acc_without_tl, loss_without_tl = scratch_run_reuse(configs, dataloaders_dict, k)
        # print("acc_without_tl, loss_without_tl", acc_without_tl, loss_without_tl)
        # construct the target_label "tar_label_imbak" according to ratio
        # randomly select ratio% data from the dt
        data_size_dt = tar_labels.shape[0]
        random_indices = np.random.choice(data_size_dt, int(data_size_dt * rat), replace=False)
        tar_features_semi = tar_features[random_indices, :]
        tar_label_semi = tar_labels[random_indices]
        # L2_norm
        src_features_norm = src_features / np.linalg.norm(src_features, axis=1, keepdims=True)
        tar_features_semi_norm = tar_features_semi / np.linalg.norm(tar_features_semi, axis=1, keepdims=True)
        src_labels_norm = src_labels / np.linalg.norm(src_labels)
        tar_labels_semi_norm = tar_label_semi / np.linalg.norm(tar_label_semi) # (461, 20)
        # compute lipz constant
        num_samples = len(tar_labels_semi_norm)
        # pdb.set_trace()
        constant = np.linalg.norm(tar_features_semi_norm.T.dot(tar_features_semi_norm), ord=2) + np.linalg.norm((tar_labels_semi_norm.T).dot(tar_features_semi_norm), ord=2)
        lipschitz_constant = constant/num_samples
        print("lipschitz_constant:", lipschitz_constant)
        alph, ot2_bound, my_bound, Es, time = bound_estimation_semi(
                        mse_ds, src_features_norm,
                        src_labels_norm,
                        tar_features_semi_norm,
                        tar_labels_semi_norm,
                        lipschitz_constant)
        ot2_bound_all.append(ot2_bound)
        my_bound_all.append(my_bound)
        # lipz_all.append(lipschitz_constant)
        alpha.append(alph)
        # acc.append(acc_without_tl)
        # loss.append(loss_without_tl)
        with open(os.path.join('./tran_metrics_smi/proposed',
                  "trans_ds_" + ds + "_dt_" + dt + ".txt"),
                   'a+', encoding='utf-8') as fw:
            fw.write("ratio for semi supervised setting: %f\t" % rat)
            fw.write("ot2_bound: %.4f\t" % ot2_bound)
            fw.write("my_bound: %.4f\t" % my_bound)
            fw.write("lipschitz_constant:\t" % lipschitz_constant)
            fw.write("alpha: %.4f\t" % alph)
            fw.write("Es: %.4f\t" % Es)
            fw.write("time: %.4f\t" % time)
            fw.write("\n")
            fw.close()
    # write files
    with open(os.path.join('./tran_metrics_smi/proposed', "esti_hist_ds_" + ds + "_dt_" + dt + ".txt"), 'a+',
                      encoding='utf-8') as fw:
        fw.write("\n")
        fw.write("my_bound_all:\t")
        for i in my_bound_all:
            fw.write(format(i, '.4f') + '\t')
        fw.write("ot2_bound_all:\t")
        for i in ot2_bound_all:
            fw.write(format(i, '.4f') + '\t')
        fw.write("alpha:\t")
        for i in alpha:
            fw.write(format(i, '.4f') + '\t')
        fw.close()
    return np.array(ot2_bound_all), np.array(my_bound_all)

def data_for_bound(data_name):
    trainset = Turbofandataset(mode='train',
                                  dataset='./datasets/CMAPSSData/train_' + data_name + '_normed.txt')
    train_loader = DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = Turbofandataset(mode='test',
                                 dataset='./datasets/CMAPSSData/test_' + data_name + '_normed.txt',
                                 rul_result='./datasets/CMAPSSData/RUL_' + data_name + '.txt')
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=2)
    print('dataset load successfully!')
    num_fea = Turbofandataset.__get_feature_num__(trainset)
    # pdb.set_trace()
    # fea = np.concatenate((trainset.x, testset.x), axis=0)
    # label = np.concatenate((trainset.y, testset.y), axis=0)

    return train_loader, test_loader, num_fea

def cal_corr(empi_path, b_c, b_ot2):
    # load the pre-saved eval_from_scratch.txt
    rmse = []
    with open(empi_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split("\t")
            if parts == ['\n']:
                break
            else:
                rmse.append(float(parts[0]))

    print("corr between mse and b_c_metric")
    calculate_cor(rmse, b_c)
    print("corr between mse and b_ot2_metric")
    calculate_cor(rmse, b_ot2)


if __name__ == "__main__":
    # configs = Config()
    ds = 'FD001'
    dt = 'FD003'
    rs = 14.268
    # rs = 23.3528   # ds FD004
    empi_path = "./tran_metrics_smi/others/empi_1_to_3.txt"
    ot2_bound_all, my_bound_all = main_with_N_K(ds, dt, rs)
    cal_corr(empi_path, my_bound_all, ot2_bound_all)
