# This script illustrates the stability of learning vITL
# for emotion transfer in the presence of missing data
# Runtime ~1h on laptop
# ----------------------------------
# Imports
# ----------------------------------
import os
import torch
import matplotlib.pyplot as plt
import sys
import importlib

if importlib.util.find_spec('torch_itl') is None:
    path_to_lib = os.getcwd()[:-23]
    sys.path.append(path_to_lib)

from torch_itl.sampler import CircularEmoSampler
from torch_itl.model import DecomposableIdentity
from torch_itl.kernel import Gaussian
from torch_itl.estimator import EmoTransfer
from torch_itl.datasets import get_data_landmarks
# %%
# ----------------------------------
# Reading input/output data
# ----------------------------------
path_to_rafd = '../../torch_itl/datasets/Rafd_Aligned/Rafd_LANDMARKS'
path_to_kdef = '../../torch_itl/datasets/KDEF_Aligned/KDEF_LANDMARKS'
# test of import
data_train, data_test = get_data_landmarks('KDEF', path_to_kdef)
n, m, nf = data_train.shape
print('Testing import, data dimensions:', n, m, nf)
# %%
# ----------------------------------
# Defining our model
# ----------------------------------
print('Defining the model')
# define Landmarks kernel
gamma_inp = 0.07
kernel_input = Gaussian(gamma_inp)
# define emotion kernel
gamma_out = 0.4
kernel_output = Gaussian(gamma_out)
# define functional model
model = DecomposableIdentity(kernel_input, kernel_output, nf)
# define emotion sampler
sampler = CircularEmoSampler()
# define regularization
lbda = 2e-5
# define the emotion transfer estimator
est = EmoTransfer(model, lbda,  sampler, inp_emotion='joint')
#%%
# ----------------------------------
# Learning in the presence of missing data -KDEF
# ----------------------------------
print('Learning with missing data KDEF')
# number of random masks of each size
n_loops = 4
# results tensor
test_losses_kdef = torch.zeros(10, n_loops, n)

for kfold in range(10):
    get_data_landmarks('KDEF', path_to_kdef, kfold=kfold)
    mask_list = [torch.randperm(n * m).reshape(n, m) for j in range(n_loops)]
    for j in range(n_loops):
        mask_level = mask_list[j]
        for i in torch.arange(n * m)[::7]:
            mask = (mask_level >= i)
            est.fit_partial(data_train, mask)
            test_losses_kdef[kfold, j, i // 7] = est.risk(data_test)
    print('done with kfold ', kfold)
# %%
#torch.save(test_losses_kdef, 'kdef_partial.pt')
# %%
# ----------------------------------
# Learning in the presence of missing data -Rafd
# ----------------------------------
print('Learning with missing data RaFD')
# number of random masks of each size
n_loops = 4
# results tensor
n = 61
test_losses_rafd = torch.zeros(10, n_loops, n)

for kfold in range(1, 11):
    get_data_landmarks('RaFD', path_to_rafd, kfold=kfold)
    n, m, _ = data_train.shape
    mask_list = [torch.randperm(n * m).reshape(n, m) for j in range(n_loops)]
    for j in range(n_loops):
        mask_level = mask_list[j]
        for i in torch.arange(n * m)[::7]:
            mask = (mask_level >= i)
            est.fit_partial(data_train, mask)
            test_losses_rafd[kfold - 1, j, i // 7] = est.risk(data_test)
#%%
#torch.save(test_losses_rafd, 'rafd_partial.pt')
#%%
idx_kdef = torch.arange(test_losses_kdef.shape[2]*m)[::7].float() / test_losses_kdef.shape[2] / m
idx_rafd = torch.arange(test_losses_rafd.shape[2]*m)[::7].float() / n/m
#%%
mean_kdef = test_losses_kdef.mean(1).mean(0)
max_kdef , _ = test_losses_kdef.mean(1).max(axis=0)
min_kdef , _ = test_losses_kdef.mean(1).min(axis=0)

mean_rafd = test_losses_rafd.mean(1).mean(0)
max_rafd , _ = test_losses_rafd.mean(1).max(axis=0)
min_rafd , _ = test_losses_rafd.mean(1).min(axis=0)
#%%
plt.figure()
plt.xlabel("% of missing data")
plt.ylabel("$\log_{10}$ Test MSE")
plt.plot(idx_kdef, torch.log(mean_kdef), c='black', label='KDEF mean', marker=',')
plt.plot(idx_kdef, torch.log(min_kdef), c='black', label='KDEF min-max', linestyle='--')
plt.plot(idx_kdef, torch.log(max_kdef), c='black', linestyle='--')
plt.plot(idx_rafd, torch.log(mean_rafd), c='grey', label='RaFD mean', marker=',')
plt.plot(idx_rafd, torch.log(min_rafd), c='grey', label='RaFD min-max', linestyle='--')
plt.plot(idx_rafd, torch.log(max_rafd), c='grey', linestyle='--')
plt.legend(loc='upper left')
plt.savefig('partial_observation.pdf')
plt.show()
