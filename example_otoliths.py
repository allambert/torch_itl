import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch_itl import model, sampler, cost, kernel, estimator

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

X_train.shape
n_h = 80
d_out = 50
model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], n_h),
    torch.nn.ReLU(),
    torch.nn.Linear(n_h, n_h),
    torch.nn.Linear(n_h, d_out),
)
optim_params = dict(lr=0.001, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)

kernel_input = kernel.GaussianRFF(X_train.shape[1],50,3)
kernel_output = kernel.GaussianRFF(1,30,4)
itl_model = model.RFFModel(kernel_input, kernel_output)
cost_function = cost.ploss_with_crossing(0.01)
lbda = 0.001
sampler_ = sampler.LinearSampler(0.05, 0.95, epsilon=0)
sampler_.m = 25

itl_estimator = estimator.ITLEstimator(itl_model,
                                       cost_function, lbda, sampler_)

itl_estimator.fit_alpha(X_train, Y_train, n_epochs=20,
                        lr=0.001, line_search_fn='strong_wolfe')

# %%

plt.figure()
plt.plot(itl_estimator.losses)
plt.show()

# %%
