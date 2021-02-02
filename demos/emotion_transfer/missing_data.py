# ----------------------------------
# Imports
# ----------------------------------
import os
import json
import time
import torch
import matplotlib.pyplot as plt
import sys
import importlib

if importlib.util.find_spec('torch_itl') is None:
    path_to_lib =!pwd
    path_to_lib = path_to_lib[0][:-23]
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
# Learning in the presence of missing data
# ----------------------------------
