import torch
import numpy as np
from torch import optim as optim
from turbofandataset import Turbofandataset
from torch.utils.data import DataLoader
from model_lstm import LSTM1
from train_lstm import Trainer_lstm


if __name__ == '__main__':
    max_iter = 1

    trainset = Turbofandataset(mode='train',
                               dataset='./datasets/CMAPSSData/train_FD002_normed.txt')
    train_loader = DataLoader(dataset=trainset, batch_size=100, shuffle=True, num_workers=4)
    num_fea = Turbofandataset.__get_feature_num__(trainset)
    testset = Turbofandataset(mode='test',
                              dataset='./datasets/CMAPSSData/test_FD002_normed.txt',
                              rul_result='./datasets/CMAPSSData/RUL_FD002.txt')
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4)
    print('dataset load successfully!')

    best_score_list = []
    best_RMSE_list = []
    for iteration in range(max_iter):
        print('---Iteration: {}---'.format(iteration + 1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTM1(input_size=num_fea, hidden_size=96, num_layers=4, device=device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 32

        trainer = Trainer_lstm(model=model,
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
    np.savetxt('./ds_results_lstm/{}_result.txt'.format(trainer.prefix), result, fmt='%.4f')
