import csv
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress


# pdb.set_trace()
def draw(metric, acc):
    plt.style.use('ggplot')
    slope, intercept, r, p, stderr = linregress(acc, metric)
    line = f'Regression line: acc={intercept:.2f}+{slope:.2f}avg_metric, r={r:.2f}'
    fig, ax = plt.subplots()
    ax.plot(metric, acc, linewidth=0, marker='s', label='Data points')
    ax.plot(metric, intercept + slope * metric, label=line)
    ax.set_xlabel('metric')
    ax.set_ylabel('acc')
    ax.legend(facecolor='white')
    plt.show()


def calculate_cor(acc, metric):
    pccs = pearsonr(acc, metric)
    print("pearson:", format(pccs[0], '.4f'), '\t', pccs[1])
    kendall = kendalltau(acc, metric)
    print("kendall", format(kendall.correlation, '.4f'), '\t', kendall.pvalue)
    pearsonrpccs = spearmanr(acc, metric)
    print("spearmanr", format(pearsonrpccs.correlation, '.4f'), '\t', pearsonrpccs.pvalue)
    return format(pccs[0], '.4f'), format(kendall.correlation, '.4f'), format(pearsonrpccs.correlation, '.4f')



"""trainData = pd.read_csv('corr_f1_to_f2.csv', sep = ',')
# trainData = pd.read_csv('corr_for_compare_smi.csv')
# for 2nd sampling
# loss = np.array(trainData.iloc[:, 7:13])
# acc = np.array(trainData.iloc[:, 1:7])
# c_metric = np.array(trainData.iloc[:, 13:19])
# lipz_c_metric = np.array(trainData.iloc[:, 19:25])
mse = np.array(trainData.iloc[:, 1])
b_c = np.array(trainData.iloc[:, 2])
b_ot2 = np.array(trainData.iloc[:, 3])
print("corr between mse and b_c_metric")
calculate_cor(mse, b_c)
print("corr between mse and b_ot2_metric")
calculate_cor(mse, b_ot2)"""

if __name__ == "__main__":
    trainData = pd.read_csv('empi_acc.csv')
    empi_rmse = np.array(trainData.iloc[:, 0:3])
    my_bound = np.array(trainData.iloc[:, 3:6])
    for i in range(3):

        print("accuracy between empi_rmse and my_bound")
        calculate_cor(empi_rmse[:, i], my_bound[:,i])
