"""Cross validation script for tuning gamma_inp, gamma_out, and lbda.
Can take a while."""

import os
import torch
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
# Please replace those values with the right path to
# the extracted landmarks on your computer.
# See utils/README.md
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
# Cross Validation Loop KDEF
# ----------------------------------
print('Cross validation loop for KDEF')
input_type_list = [i for i in range(m)] + ['joint']
n_lbda = 10
n_gamma_inp = 10
n_gamma_out = 10
lbda_list = torch.logspace(-6, -3, n_lbda)
gamma_inp_list = torch.logspace(-1.6, -0.5, n_gamma_inp)
gamma_out_list = torch.logspace(-1.5, 0, n_gamma_out)
risk_kdef = torch.zeros(m+1, 10, 6, n_lbda, n_gamma_inp, n_gamma_out)

for n_input, input_type in enumerate(input_type_list):
    print('Computing for input_type =', input_type, ', kfold = ', end=" ")
    est.input_type = input_type
    for kfold in range(10):
        if kfold == 9:
            print(kfold)
        else:
            print(kfold, end=" ")
        data, data_test = get_data_landmarks('KDEF', path_to_kdef, kfold=kfold)
        n = data.shape[0]
        mask = torch.randperm(n)
        for kval in range(6):
            mask_train = (mask >= 21*(kval+1)) + (21*kval > mask)
            mask_val = (mask < 21*(kval+1))*(mask >= 21*kval)
            data_train = data[mask_train]
            data_val = data[mask_val]
            for i, lbda in enumerate(lbda_list):
                est.lbda = lbda
                for j, gamma_inp in enumerate(gamma_inp_list):
                    est.model.kernel_input.gamma = gamma_inp
                    for k, gamma_out in enumerate(gamma_out_list):
                        est.model.kernel_output.gamma = gamma_out
                        est.fit(data_train)
                        risk_kdef[n_input, kfold, kval,
                                  i, j, k] = est.risk(data_val)
# %%
print('Computing argmin')
gamma_inp_argmin_kdef = torch.ones(m+1, 10)
gamma_out_argmin_kdef = torch.ones(m+1, 10)
lbda_argmin_kdef = torch.ones(m+1, 10)

for n_input in range(m+1):
    for kfold in range(10):
        t_kdef = risk_kdef[n_input, kfold].mean(0).argmin()
        gamma_out_argmin_kdef[n_input,
                              kfold] = gamma_out_list[t_kdef % n_gamma_out]
        gamma_inp_argmin_kdef[n_input, kfold] = gamma_inp_list[(
            t_kdef//n_gamma_out) % n_gamma_inp]
        lbda_argmin_kdef[n_input, kfold] = lbda_list[(
            t_kdef//n_gamma_out)//n_gamma_inp]
# %%
print('Computing test risks with best parameters')
test_risks_kdef = torch.zeros(m+1, 10)

for n_input, input_type in enumerate(input_type_list):
    for kfold in range(10):
        est.input_type = input_type
        est.lbda_reg = lbda_argmin_kdef[n_input, kfold]
        est.model.kernel_input.gamma = gamma_inp_argmin_kdef[n_input, kfold]
        est.model.kernel_output.gamma = gamma_out_argmin_kdef[n_input, kfold]
        data_train, data_test = get_data_landmarks(
            'KDEF', path_to_kdef, kfold=kfold)
        est.fit(data_train)
        test_risks_kdef[n_input, kfold] = est.risk(data_test)

test_means_kdef = test_risks_kdef.mean(1)
test_stds_kdef = (
    (test_risks_kdef - test_means_kdef.view(-1, 1))**2).mean(1).sqrt()
# %%
print("Mean Test Errors KDEF:", test_means_kdef)
print("Mean Test Stds KDEF:", test_stds_kdef)
# %%
torch.save(lbda_argmin_kdef, 'KDEF_lbdas.pt')
torch.save(gamma_inp_argmin_kdef, 'KDEF_gamma_inp.pt')
torch.save(gamma_out_argmin_kdef, 'KDEF_gamma_out.pt')
# %%
# ----------------------------------
# Cross Validation Loop RaFD
# ----------------------------------
print('Cross validation loop for RaFD:')
input_type_list = [i for i in range(m)] + ['joint']
n_lbda = 10
n_gamma_inp = 10
n_gamma_out = 10
lbda_list = torch.logspace(-6, -3, n_lbda)
gamma_inp_list = torch.logspace(-1.6, -0.5, n_gamma_inp)
gamma_out_list = torch.logspace(-1.5, 0, n_gamma_out)
risk_rafd = torch.zeros(m+1, 10, 10, n_lbda, n_gamma_inp, n_gamma_out)

for n_input, input_type in enumerate(input_type_list):
    print('Computing for input_type =', input_type, ', kfold = ', end=" ")
    est.input_type = input_type
    for kfold in range(1, 11):
        if kfold == 10:
            print(kfold - 1)
        else:
            print(kfold - 1, end=" ")
        data, data_test = get_data_landmarks('RaFD', path_to_rafd, kfold=kfold)
        n = data.shape[0]
        mask = torch.randperm(n)
        for kval in range(10):
            mask_train = (mask >= 6*(kval+1)) + (6*kval > mask)
            mask_val = (mask < 6*(kval+1))*(mask >= 6*kval)
            data_train = data[mask_train]
            data_val = data[mask_val]
            for i, lbda in enumerate(lbda_list):
                est.lbda = lbda
                for j, gamma_inp in enumerate(gamma_inp_list):
                    est.model.kernel_input.gamma = gamma_inp
                    for k, gamma_out in enumerate(gamma_out_list):
                        est.model.kernel_output.gamma = gamma_out
                        est.fit(data_train)
                        risk_rafd[n_input, kfold-1, kval,
                                  i, j, k] = est.risk(data_val)
# %%
print('Computing argmin')
gamma_inp_argmin_rafd = torch.ones(m+1, 10)
gamma_out_argmin_rafd = torch.ones(m+1, 10)
lbda_argmin_rafd = torch.ones(m+1, 10)

for n_input in range(m+1):
    for kfold in range(10):
        t_rafd = risk_rafd[n_input, kfold].mean(0).argmin()
        gamma_out_argmin_rafd[n_input,
                              kfold] = gamma_out_list[t_rafd % n_gamma_out]
        gamma_inp_argmin_rafd[n_input, kfold] = gamma_inp_list[(
            t_rafd//n_gamma_out) % n_gamma_inp]
        lbda_argmin_rafd[n_input, kfold] = lbda_list[(
            t_rafd//n_gamma_out)//n_gamma_inp]
# %%
print('Computing test risk for best parameters')
test_risks_rafd = torch.zeros(m+1, 10)

for n_input, input_type in enumerate(input_type_list):
    est.input_type = input_type
    for kfold in range(1, 11):
        est.lbda = lbda_argmin_rafd[n_input, kfold-1]
        est.model.kernel_input.gamma = gamma_inp_argmin_rafd[n_input, kfold-1]
        est.model.kernel_output.gamma = gamma_out_argmin_rafd[n_input, kfold-1]
        data_train, data_test = get_data_landmarks(
            'RaFD', path_to_rafd, kfold=kfold)
        est.fit(data_train)
        test_risks_rafd[n_input, kfold-1] = est.risk(data_test)

test_means_rafd = test_risks_rafd.mean(1)
test_stds_rafd = (
    (test_risks_rafd - test_means_rafd.view(-1, 1))**2).mean(1).sqrt()
# %%
print("Mean Test Errors RaFD:", test_means_rafd)
print("Mean Test Stds RaFD:", test_stds_rafd)
# %%
torch.save(lbda_argmin_rafd, 'Rafd_lbdas.pt')
torch.save(gamma_inp_argmin_rafd, 'Rafd_gamma_inp.pt')
torch.save(gamma_out_argmin_rafd, 'Rafd_gamma_out.pt')
