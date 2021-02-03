# This script illustrate that we can reduce the dimensionality of our model
# when using vITL by choosing A to be a non-invertible well chosen
# matrix, built upon the empirical covariance matrix
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
# %%
# ----------------------------------
# Fitting fewer coefficients with non invertible matrix A -- KDEF
# ----------------------------------
print('Computing losses KDEF')
losses_kdef = torch.zeros(10, nf)
for kfold in range(10):
    data_train, data_test = get_data_landmarks('KDEF', path_to_kdef, kfold=kfold)
    for r in range(nf):
        est.fit_dim_red(data_train, r+1)
        losses_kdef[kfold, r] = est.risk(data_test)
# %%
# ----------------------------------
# Fitting fewer coefficients with non invertible matrix A -- Rafd
# ----------------------------------
print('Computing losses RaFD')
losses_rafd = torch.zeros(10, nf)
for kfold in range(1,11):
    data_train, data_test = get_data_landmarks('RaFD', path_to_rafd, kfold=kfold)
    for r in range(nf):
        est.fit_dim_red(data_train, r+1)
        losses_rafd[kfold-1, r] = est.risk(data_test)
# %%
# ----------------------------------
# Saving the results
# ----------------------------------
#torch.save(losses_kdef,'dim_red_kdef.pt')
#torch.save(losses_rafd,'dim_red_rafd.pt')
# %%
# ----------------------------------
# Plotting
# ----------------------------------
std_rafd = ((losses_rafd - losses_rafd.mean(0))**2).mean(0).sqrt()
std_kdef = ((losses_kdef - losses_kdef.mean(0))**2).mean(0).sqrt()
plt.figure()
plt.xlabel('Rank of $\mathbf{A}$')
plt.ylabel('Test MSE')
plt.plot(losses_kdef.mean(0), c='black', label='KDEF mean', marker=',')
plt.plot(losses_kdef.mean(0) + std_kdef, c='black', label='KDEF mean $\pm \sigma$', linestyle='--')
plt.plot(losses_kdef.mean(0) - std_kdef, c='black', linestyle='--')
plt.plot(losses_rafd.mean(0), c= 'grey', label='RaFD mean', marker=',')
plt.plot(losses_rafd.mean(0) + std_rafd, c= 'grey', label='RaFD mean $\pm \sigma$', linestyle='--')
plt.plot(losses_rafd.mean(0) - std_rafd, c= 'grey', linestyle='--')

plt.legend(loc='upper right')
plt.savefig('dim_red.pdf')
plt.show()
