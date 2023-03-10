import numpy as np
import pandas as pd

from compute_corr import calculate_cor


def acc(my_tran, empi_tran):
    num_task = len(my_tran)
    # prediction is positive but ground_truth is negative
    num_pos_neg = 0
    num_neg_pos = 0
    num_neg_neg = 0
    for i in range(num_task):
        if my_tran[i] > 0 > empi_tran[i]:
            num_neg_pos +=1
        elif my_tran[i] < 0  and empi_tran[i] < 0:
            num_neg_neg +=1
        elif my_tran[i] < 0 < empi_tran[i]:
            num_pos_neg +=1
    if num_neg_pos + num_neg_neg>0:
        TNR = num_neg_neg / (num_neg_pos + num_neg_neg)
    else:

        TNR = 'error'

    num_pos_pos = num_task - num_neg_neg - num_pos_neg - num_neg_pos
    return num_neg_neg, num_pos_neg, num_neg_pos, num_pos_pos, TNR


if __name__ == "__main__":
    trainData = pd.read_csv('empi_acc.csv')
    my_trans = np.array(trainData.iloc[:, 0])
    empi_trans = np.array(trainData.iloc[:, 1])

    print("accuracy between my_trans and empi_trans")
    TN, FN, FP, TP, TNR = acc(my_trans, empi_trans)
    print("TNR", TNR)
    print("TP", TP)
    print("FN", FN)
    print("FP", FP)
    print("TN", TN)





