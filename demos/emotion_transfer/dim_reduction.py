# This script illustrate that we can reduce the dimensionality of our model
# when using vITL by choosing A to be a non-invertible well chosen
# matrix, built upon the empirical covariance matrix
# Runtime ~25 min on laptop
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
# %%
# ----------------------------------
# Reading input/output data
# ----------------------------------
dataset = 'KDEF'  # KDEF or Rafd
inc_emotion = True  # bool to include (0,0) in emotion embedding
use_facealigner = True  # bool to use aligned faces (for 'Rafd' - set to true)

data_path = os.path.join('../../' + './datasets', dataset +
                         '_Aligned', dataset + '_LANDMARKS')  # set data path
data_emb_path = os.path.join(
    '../../' + './datasets', dataset + '_Aligned', dataset + '_facenet')  # set data path


def get_data(dataset, kfold=0):
    if dataset == 'Rafd':
        # dirty hack only used to get Rafd speaker ids, not continuously ordered
        data_csv_path = '/Users/alambert/Recherche/ITL/code/Rafd.csv'
    affect_net_csv_path = ''  # to be set if theta_type == 'aff'
    # store all experiments in this output folder
    output_folder = './LS_Experiments/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    #print('Reading data')
    if use_facealigner:
        input_data_version = 'facealigner'
        if dataset == 'KDEF':
            from datasets.datasets import kdef_landmarks_facealigner, kdef_landmarks_facenet
            x_train, y_train, x_test, y_test, train_list, test_list = \
                kdef_landmarks_facealigner(data_path, inp_emotion='NE',
                                           inc_emotion=inc_emotion, kfold=kfold)
        elif dataset == 'Rafd':
            from datasets.datasets import rafd_landmarks_facealigner
            x_train, y_train, x_test, y_test, train_list, test_list = \
                rafd_landmarks_facealigner(data_path, data_csv_path, inp_emotion='angry',
                                           inc_emotion=inc_emotion, kfold=kfold)
    else:
        from datasets.datasets import import_kdef_landmark_synthesis
        input_data_version = 'aligned2'
        x_train, y_train, x_test, y_test = import_kdef_landmark_synthesis(
            dtype=input_data_version)
    return (y_train, y_test)


# test of import
data_train, data_test = get_data(dataset, 10)
n, m, nf = data_train.shape
print('data dimensions', n, m, nf)

# %%
# ----------------------------------
# Defining our model
# ----------------------------------

# define Landmarks kernel
gamma_inp = 0.07
kernel_input = Gaussian(gamma_inp)

# define emotion kernel
gamma_out = 0.4
kernel_output = Gaussian(gamma_out)

# define functional model
model = DecomposableIdentity(kernel_input, kernel_output, nf)

# define emotion sampler
sampler = CircularEmoSampler(dataset=dataset)

# define regularization
lbda = 2e-5

# defining the model
est = EmoTransfer(model, lbda,  sampler)

# %%
# ----------------------------------
# Fitting fewer coefficients with non invertible matrix A -- KDEF
# ----------------------------------
losses_kdef = torch.zeros(10, nf)
for kfold in range(10):
    data_train, data_test = get_data(dataset, kfold)
    for r in range(nf):
        est.fit_dim_red(data_train, r+1)
        losses_kdef[kfold, r] = est.risk(data_test)


# %%
# ----------------------------------
# Fitting fewer coefficients with non invertible matrix A -- Rafd
# ----------------------------------
dataset = 'Rafd'
est.sampler.dataset = dataset
data_path = os.path.join('../../' + './datasets', dataset +
                         '_Aligned', dataset + '_LANDMARKS')  # set data path
data_emb_path = os.path.join(
    '../../' + './datasets', dataset + '_Aligned', dataset + '_facenet')  # set data path

losses_rafd = torch.zeros(10, nf)
for kfold in range(1,11):
    data_train, data_test = get_data(dataset, kfold)
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
