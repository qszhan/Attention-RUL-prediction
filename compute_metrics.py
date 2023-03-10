# coding=utf-8

"""
Compute different transferability metrics.
"""

import os
import pdb
import pickle
import random
import time
import numpy as np
import scipy.optimize
import scipy.special
import sklearn.decomposition
import sklearn.mixture
import sklearn.neighbors
import sklearn.svm
import torch
import argparse
# import geomloss








def calculate_hscore(features_np_all, label_np_all):
    """Calculate hscore."""
    starttime = time.time()
    # label_np_all = one_hot(label_np_all)
    mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
    features_np_all -= mean_feature  # [n, k]
    d = features_np_all.shape[1]
    covf = np.matmul(
        features_np_all.transpose(), features_np_all) + np.eye(d) * 1e-6

    yf = np.matmul(label_np_all.transpose(), features_np_all)  # [v, k]
    sumy = np.sum(label_np_all, axis=0) + 1e-10  # [v]
    fcy = yf / np.expand_dims(sumy, axis=-1)  # [v, k]
    covfcy = np.matmul(yf.transpose(), fcy)

    hscore = np.trace(np.matmul(np.linalg.pinv(covf), covfcy))
    endtime = time.time()
    return hscore, (endtime - starttime)



def calculate_nce(source_label: np.ndarray, target_label: np.ndarray):
    """

    :param source_label: shape [N], elements in [0, C_s), often got from taken argmax from pre-trained predictions
    :param target_label: shape [N], elements in [0, C_t)
    :return:
    """
    starttime = time.time()
    C_t = int(np.max(target_label) + 1)  # the number of target classes
    C_s = int(np.max(source_label) + 1)  # the number of source classes
    N = len(source_label)
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for the joint distribution, shape [C_t, C_s]
    for s, t in zip(source_label, target_label):
        s = int(s)
        t = int(t)
        joint[t, s] += 1.0 / N
    p_z = joint.sum(axis=0, keepdims=True)  # shape [1, C_s]
    p_target_given_source = (joint / p_z).T  # P(y | z), shape [C_s, C_t]
    mask = p_z.reshape(-1) != 0  # valid Z, shape [C_s]
    p_target_given_source = p_target_given_source[mask] + 1e-20  # remove NaN where p(z) = 0, add 1e-20 to avoid log (0)
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)  # shape [C_s, 1]
    conditional_entropy = np.sum(entropy_y_given_z * p_z.reshape((-1, 1))[mask]) # scalar
    endtime = time.time()
    return -conditional_entropy, (endtime - starttime)


