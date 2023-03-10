import pdb

import torch
import numpy as np
from torch import optim as optim
from turbofandataset import Turbofandataset
from torch.utils.data import DataLoader
from model import Model
from train import Trainer

if __name__ == '__main__':
    max_iter = 1
    # pdb.set_trace()
    trainset = Turbofandataset(mode='train',
                               dataset='./datasets/CMAPSSData/train_FD002qs_normed.txt')
    train_loader = DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
    num_fea = Turbofandataset.__get_feature_num__(trainset)
    testset = Turbofandataset(mode='test',
                              dataset='./datasets/CMAPSSData/test_FD002_normed.txt',
                              rul_result='./datasets/CMAPSSData/RUL_FD002.txt')
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=2)
    print('dataset load successfully!')
    # pdb.set_trace()

    best_score_list = []
    best_RMSE_list = []
    for iteration in range(max_iter):
        print('---Iteration: {}---'.format(iteration + 1))
        model = Model(num_fea)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 150
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(model=model,
                          model_optimizer=optimizer,
                          print_every=50,
                          epochs=epochs,
                          device=device,
                          prefix='FD002')
        best_score, best_RMSE = trainer.train(train_loader, test_loader, iteration)
        best_score_list.append(best_score)
        best_RMSE_list.append(best_RMSE)

    best_score_list = np.array(best_score_list)
    best_RMSE_list = np.array(best_RMSE_list)
    result = np.concatenate((best_score_list, best_RMSE_list)).reshape(2, max_iter)
    np.savetxt('./ds_results/{}_result.txt'.format(trainer.prefix), result, fmt='%.4f')
