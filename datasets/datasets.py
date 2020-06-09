import numpy as np
import torch
import pandas as pd

from .synthetic_quantile import toy_data_quantile

def import_data_otoliths():

    filename_prefix = './datasets/otoliths/train/Train_fc_'
    filename_suffix = '.npy'
    n = 3780
    X_train = np.zeros((n, 64 * 64))
    for i in range(n):
        X_train[i] = np.load(filename_prefix + str(i) + filename_suffix)
    training_info = pd.read_csv('./datasets/otoliths/train/training_info.csv')
    Y_train = training_info['Ground truth'].values
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()

    filename_prefix = './datasets/otoliths/test/Test_fc_'
    filename_suffix = '.npy'
    n = 165
    X_test = np.zeros((n, 64 * 64))
    for i in range(n):
        X_test[i] = np.load(filename_prefix + str(i) + filename_suffix)
    testing_info = pd.read_csv('./datasets/otoliths/test/Testing_info.csv')
    Y_test = testing_info['Ground truth'].values
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()
    return X_train, Y_train, X_test, Y_test

def import_data_toy_quantile(*args):
    x_train, y_train, probs = toy_data_quantile(*args)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    probs = torch.Tensor(probs)
    return x_train, y_train, probs
