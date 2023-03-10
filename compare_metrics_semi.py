import heapq
import random
import time

import numpy as np
import ot
import torch
import math
from scipy.spatial.distance import cdist
import pdb
import os
from collections import Counter


from LogME import LogME
from compute_metrics import calculate_nce, calculate_hscore
from compute_corr import calculate_cor

# assert torch.cuda.is_available(), "GPU not available"
# print(torch.cuda.is_available())
from sklearn import preprocessing
from torch import optim, nn
from torchvision import datasets, models, transforms

# import geomloss


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_metrics(empi_path, ds, dt):
    # step 1: load the saved features and labels for the Ds and Dt"""
    sre_data = torch.load(os.path.join('./features/ds_' + ds, ds + ".pth"))
    src_features = sre_data['features']
    src_labels = sre_data['labels']
    tar_data = torch.load(os.path.join('./features/ds_' + ds, dt + ".pth"))
    tar_features = tar_data['features']
    tar_labels = tar_data['labels']
    metric_hscore = []
    # metric_nce = []
    metric_logme = []
    # load the pre-saved eval_from_scratch.txt
    best_rmse = []
    with open(empi_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split("\t")
            if parts == ['\n']:
                break
            else:
                best_rmse.append(float(parts[0]))
    for rat in np.arange(0.001, 0.05, 0.001):
        # for rat in [0.05, 0.1, 0.15]:
        data_size_dt = tar_labels.shape[0]
        random_indices = np.random.choice(data_size_dt, int(data_size_dt * rat), replace=False)
        tar_features_semi = tar_features[random_indices, :]
        tar_label_semi = tar_labels[random_indices]
        # L2_norm
        src_features_norm = src_features / np.linalg.norm(src_features, axis=1, keepdims=True)
        tar_features_semi_norm = tar_features_semi / np.linalg.norm(tar_features_semi, axis=1, keepdims=True)
        src_labels_norm = src_labels / np.linalg.norm(src_labels)
        tar_labels_semi_norm = tar_label_semi / np.linalg.norm(tar_label_semi)  # (461, 20)
        # nce_transability, time = calculate_nce(src_labels_norm, tar_labels_semi_norm)
        logme = LogME(regression=True)
        logme_transability = logme.fit(tar_features_semi_norm, tar_labels_semi_norm)
        hscore_transability, time = calculate_hscore(tar_features_semi_norm, tar_labels_semi_norm)
        metric_hscore.append(hscore_transability)
        # metric_nce.append(nce_transability)
        metric_logme.append(logme_transability)
        # print("metric_nce", metric_nce)

    print("metric_logme", metric_logme)
    print("metric_h_score", metric_hscore)
    with open(os.path.join("tran_metrics_smi/others", "esti_ds_" + ds + "_dt_" + dt +".txt"), 'a+',
                  encoding='utf-8') as fw:
        fw.write("\n")
        fw.write("logme:\t")
        # pdb.set_trace()
        for i in metric_logme:
            fw.write(format(i, '.4f') + '\t')
        fw.write("hscore:\t")
        for i in metric_hscore:
            fw.write(format(i, '.4f') + '\t')
        fw.close()

    return np.array(metric_logme), np.array(metric_hscore), np.array(best_rmse)



if __name__ == "__main__":
    ds = "FD001"
    dt = "FD003"
    empi_path = "./tran_metrics_smi/others/empi_1_to_3.txt"
    print("ds, dt, path", ds, dt, empi_path)
    metric_logme, metric_hscore, rmse = cal_metrics(empi_path, ds, dt)
    # print("corr between rmse and metric_nce")
    # calculate_cor(rmse, metric_nce)
    print("corr between rmse and metric_logme")
    calculate_cor(rmse, metric_logme)
    print("corr between rmse and metric_hscore")
    calculate_cor(rmse, metric_hscore)



