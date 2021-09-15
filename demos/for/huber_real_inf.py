import os
import sys
import importlib
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

if importlib.util.find_spec('torch_itl') is None:
    path_to_lib = os.getcwd()[:-10]
    sys.path.append(path_to_lib)

from torch_itl.datasets import add_global_outliers_worse
from torch_itl.estimator import RobustFOR
from torch_itl.kernel import Gaussian, LaplacianSum
from torch_itl.sampler import LinearSampler
from torch_itl.model import DecomposableIdentityScalar

# %%

print('Importing data...')
pfx = '/Users/alambert/Recherche/ITL/code/torch_itl/torch_itl/datasets/FOR/Lip'
data_filename = '/EMGmatlag.csv'
target_filename = '/lipmatlag.csv'
sampling_filename = '/tfine.csv'

data = pd.read_csv(pfx + data_filename, header=None)
target = pd.read_csv(pfx + target_filename, header=None)
sampling = pd.read_csv(pfx + sampling_filename, header=None)

# %%
X_lip = torch.from_numpy(data.values.T).float()
Y_lip = torch.from_numpy(target.values.T).float()
t = torch.from_numpy(sampling.values.reshape(-1)).float().view(-1, 1)
t = t / 0.64
n_lip = X_lip.shape[0]
print('Size of X:', X_lip.shape)
print('Size of Y:', Y_lip.shape)
print('Size of t:', t.shape)
n_o = 4
X, Y = add_global_outliers_worse(X_lip, Y_lip, n_o, 1.2, seed='fixed')
n, d = X.shape
print('New size of X:', X.shape)
# %%
colors = [cm.viridis(x) for x in torch.linspace(0, 1, n_lip)] + ['r']*n_o
color_out = 'r'
plt.figure()
plt.subplot(211)
plt.title('Input functions $(x_i)_{i=1}^n$')
for i in range(n):
    plt.plot(t, X[i], color=colors[i])
plt.subplot(212)
plt.tight_layout()
plt.title('Output functions $(y_i)_{i=1}^n$')
for i in range(n):
    plt.plot(t, Y[i], color=colors[i])
plt.show(block=None)

# %%
gamma_input = torch.linspace(5, 50, 8)[0]
gamma_output = torch.linspace(20, 80, 8)[5]
lbda = torch.logspace(-4, 0, 10)[6]
kernel_input = LaplacianSum(gamma_input)
kernel_output = Gaussian(gamma_output)
sampler = LinearSampler(0, 1)
# %%
n_kappa = 40
model_rob = DecomposableIdentityScalar(kernel_input, kernel_output)
est_rob = RobustFOR(model_rob, lbda, sampler, norm='inf')
intensity_list = [0.5, 1, 2, 5]
kappa_max_list = torch.zeros(len(intensity_list))
# %%
def mask(j, n):
    res = torch.ones(n, dtype=torch.bool)
    res[j] = False
    return(res)
# %%
est_rob.kappa = 100
n_gamma = 4
n_lbda = 4
lbda_list = torch.logspace(-4, -2, 4)
gamma_list = torch.linspace(5, 50, 4)
# res = torch.zeros(n_lbda, n_gamma, n_lip)
# for j, lbda in enumerate(lbda_list):
#     est_rob.lbda = lbda
#     for s, gamma in enumerate(gamma_list):
#         est_rob.model.kernel_output.gamma = gamma
#         for i in range(n_lip):
#             train_index = mask(i, n)
#             test_index = ~mask(i, n)
#             X_train, Y_train = X[train_index], Y[train_index]
#             X_test, Y_test = X[test_index], Y[test_index]
#             est_rob.fit_gd(X_train, Y_train, t, n_epoch=30000)
#             res[j, s, i] = est_rob.risk(X_test, Y_test, t)
#             print('Step:',j,s,i)
# %%
kernel_output.gamma = gamma_list[3]
kappa_max = est_rob.get_kappa_max('inf')
# %%
# For a single intensity
# est_rob.lbda = lbda
# est_rob.model.kernel_output.gamma = gamma
# kernel_output.gamma = gamma
# kappa_list = torch.linspace(0, 1.2*kappa_max, 40)
# res = torch.zeros(40, n_lip)
# for j, kappa in enumerate(kappa_list):
#     est_rob.kappa = kappa
#     for i in range(n_lip):
#         train_index = mask(i, n)
#         test_index = ~mask(i, n)
#         X_train, Y_train = X[train_index], Y[train_index]
#         X_test, Y_test = X[test_index], Y[test_index]
#         est_rob.fit_gd(X_train, Y_train, t, n_epoch=30000)
#         res[j, i] = est_rob.risk(X_test, Y_test, t)
#         print('Step:', j, i)
# %%
# plt.figure()
# plt.plot(kappa_list, res.mean(1))
# plt.show()
# %%
res = torch.zeros(n_kappa, n_lip, len(intensity_list))
for k, intensity in enumerate(intensity_list):
    X, Y = add_global_outliers_worse(X_lip, Y_lip, n_o, intensity,
                                     seed='fixed')
    est_rob.kappa = 100
    print('Tuning lbda parameter, intensity:', k)
    lbda, _, _ = est_rob.tune_lambda(torch.logspace(-3, -1, 5), X, Y, t,
                                     n_splits=18, fit='gd')
    est_rob.lbda = lbda
    est_rob.fit_gd(X, Y, t, n_epoch=3000, warm_start=False)
    kappa_max_list[k] = est_rob.get_kappa_max('inf')
    kappa_list = torch.linspace(0, kappa_max_list[k]*1.2, n_kappa).flip(0)
    for j in range(n_lip):
        model_rob = DecomposableIdentityScalar(kernel_input, kernel_output)
        est_rob = RobustFOR(model_rob, lbda, sampler, norm='inf')
        train_index = mask(j, n)
        test_index = ~mask(j, n)
        for i, kappa in enumerate(kappa_list):
            print('Outliers intensity:', k, 'n_split:', j, 'kappa:', i)
            est_rob.kappa = kappa
            est_rob.fit_gd(X[train_index], Y[train_index], t, tol=1e-4,
                           n_epoch=3000, warm_start=True)
            res[i, j, k] = est_rob.risk(X[test_index], Y[test_index], t)

# %%
# torch.save(res, 'huber_real_inf')
# %%
loo_err = res.mean(1).flip(0)
plt.figure()
plt.title('LOO generalization error as a function of $\kappa$', fontsize=14)
plt.xlabel('$\kappa$', fontsize=14)
plt.tick_params(labelsize=14)
plt.ylabel('LOO generalization error', fontsize=14)
colors = [cm.viridis(x) for x in torch.linspace(0, 1, len(intensity_list))]
plt.hlines(0.8, 0, 0, colors='grey',
           linestyles='--', label='Ridge Regression')
for i in range(len(intensity_list)):
    plt.plot(torch.linspace(0, kappa_max_list[i]*1.2, n_kappa),
             loo_err[:, i], color=colors[i],
             label='$s_o$='+str(intensity_list[i]))
    plt.hlines(loo_err[-1, i], kappa_max_list[i]*1.2, 16,
               color=colors[i], linestyles='--')
plt.legend(fontsize=12, loc='upper right')
plt.savefig('huber_real_inf.png', dpi=300)
# %%
intensity = intensity_list[3]
model_rob = DecomposableIdentityScalar(kernel_input, kernel_output)
est_rob = RobustFOR(model_rob, lbda, sampler, kappa=100, norm='inf')
n_lbda = 3
err = torch.zeros(n_lbda, n_lip)
lbda = torch.logspace(-2, -1, 3)[1]
X, Y = add_global_outliers_worse(X_lip, Y_lip, n_o, intensity,
                                 seed='fixed')
err = torch.zeros(n_kappa, n_lip)
kappa_list_4 = torch.linspace(0, 13, n_kappa)

# %%
for j, kappa in enumerate(kappa_list_4):
    est_rob.kappa = kappa
    for i in range(n_lip):
        train_index = mask(i, n)
        test_index = ~mask(i, n)
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]
        est_rob.fit_gd(X_train, Y_train, t, n_epoch=3000)
        err[j, i] = est_rob.risk(X_test, Y_test, t)
        print('Step:', j, i)

# %%
res.shape
loo_err.shape
loo_err[:, 0].shape
err.mean(1).shape
loo_err[:, 3] = err.mean(1)
kappa_list[3] = 13
est_rob.get_kappa_max('inf')

torch.load('huber_real_inf')
