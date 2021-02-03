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
# Learning in the presence of missing data -KDEF
# ----------------------------------
# number of random masks of each size
n_loops = 4
# results tensor
test_losses_kdef = torch.zeros(10, n_loops, n)

for kfold in range(10):
    data_train, data_test = get_data(dataset, kfold)
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
dataset = 'Rafd'
est.sampler.dataset = dataset
data_path = os.path.join('../../torch_itl/datasets', dataset +
                         '_Aligned', dataset + '_LANDMARKS')  # set data path
data_emb_path = os.path.join(
    '../../torch_itl/datasets', dataset + '_Aligned', dataset + '_facenet')  # set data path

# number of random masks of each size
n_loops = 4
# results tensor
n = 61
test_losses_rafd = torch.zeros(10, n_loops, n)

for kfold in range(1, 11):
    data_train, data_test = get_data(dataset, kfold)
    n, m, _ = data_train.shape
    mask_list = [torch.randperm(n * m).reshape(n, m) for j in range(n_loops)]
    for j in range(n_loops):
        mask_level = mask_list[j]
        for i in torch.arange(n * m)[::7]:
            mask = (mask_level >= i)
            est.fit_partial(data_train, mask)
            test_losses_rafd[kfold - 1, j, i // 7] = est.risk(data_test)
    print('done with kfold ', kfold)


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
