import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch_itl import model, sampler, cost, kernel

# %%
# Data import

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

# %%
# Conditional Quantiles computation

dtype = torch.float
device = torch.device("cpu")

kernel_input = kernel.Gaussian(0.3)
kernel_output = kernel.Gaussian(9)
cost_function = cost.ploss
lbda = 0.001
sampler_ = sampler.LinearSampler(0.1, 0.9, 10, 0)
sampler_.m = 10

itl_model = model.ITL(kernel_input, kernel_output,
                      cost_function, lbda, sampler_)

itl_model.fit(X_train, Y_train, n_epochs=60,
              lr=0.001, line_search_fn='strong_wolfe')

plt.figure()
plt.plot(itl_model.losses)
plt.show()
