import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.datasets import import_kdef_landmark_synthesis
from torch_itl import model, sampler, cost, kernel, estimator

# ----------------------------------
# Reading input/output data
# ----------------------------------
print('Reading data')
input_data_version = 'aligned2'
x_train, y_train, x_test, y_test = import_kdef_landmark_synthesis(dtype=input_data_version)
n = x_train.shape[0]
m = y_train.shape[1]
nf = y_train.shape[2]
print('data dimensions', n, m, nf)


# ----------------------------------
# Set kernel and other params
# ----------------------------------

kernel_input_learnable = False
output_var_dependence = False
save_model = True
plot_fig = True

if kernel_input_learnable:
    NE = 10  # num epochs overall
    ne_fa = 5  # num epochs fit alpha per NE
    ne_fki = 5  # num epochs fit kernel per NE

    print('defining network')
    # define input network and kernel
    n_h = 64
    d_out = 20
    model_kernel_input = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], n_h),
        torch.nn.ReLU(),
        torch.nn.Linear(n_h, d_out)
    )

    gamma_inp = 3
    optim_params = dict(lr=0.01, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)

    kernel_input = kernel.LearnableGaussian(
        gamma_inp, model_kernel_input, optim_params)
else:
    NE = 1
    ne_fa = 50
    gamma_inp = 3.0
    kernel_input = kernel.Gaussian(gamma_inp)

# define emotion kernel
gamma_out = 1.0
kernel_output = kernel.Gaussian(gamma_out)

# Define PSD matrix on output variables
if output_var_dependence:
    kernel_freq = np.load('output_kernel_matrix_kdef2.npy')
else:
    kernel_freq = np.eye(nf)

# learning rate of alpha
lr_alpha = 0.01

# ----------------------------------
# Define model
# ----------------------------------
itl_model = model.SpeechSynthesisKernelModel(kernel_input, kernel_output,
                                             kernel_freq=torch.from_numpy(kernel_freq).float())

# define cost function
cost_function = cost.speech_synth_loss
lbda = 0.001

# define emotion sampler
sampler_ = sampler.CircularSampler(data='kdef')
sampler_.m = m


itl_estimator = estimator.ITLEstimator(itl_model,
                                       cost_function, lbda, sampler_)


# ----------------------------------
# Training
# ----------------------------------

for ne in range(NE):
    itl_estimator.fit_alpha(x_train, y_train, n_epochs=ne_fa,
                        lr=lr_alpha, line_search_fn='strong_wolfe', warm_start=False)

    print(itl_estimator.losses)
    itl_estimator.clear_memory()
    if kernel_input_learnable:
        itl_estimator.fit_kernel_input(x_train, y_train, n_epochs=ne_fki)
        print(itl_estimator.model.kernel_input.losses)
        itl_estimator.model.kernel_input.clear_memory()

# ----------------------------------
# Save model and params
# -----------------------------------

if save_model:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    MODEL_DIR = './models/kdef_itl_model_' + timestr
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    PATH =  os.path.join(MODEL_DIR, 'itl_model_ckpt_' + timestr + '.pt')
    torch.save({'itl_alpha': itl_estimator.model.alpha.data,
                'itl_kernel_input': model_kernel_input.state_dict() if kernel_input_learnable else None,
                'itl_kernel_output': None}, PATH)

    PARAM_PATH = os.path.join(MODEL_DIR, 'itl_model_config_' + timestr + '.json')
    PARAM_DICT = {'Data': {'n': n, 'm': m, 'nf': nf,
                           'input_data_version': input_data_version},
                  'Kernels': {'kernel_input_learnable': kernel_input_learnable,
                              'output_var_dependence': output_var_dependence,
                              'gamma_inp': gamma_inp,
                              'gamma_out': gamma_out},
                  'Training': {'NE': NE,
                               'ne_fa': ne_fa,
                               'ne_fki': ne_fki if kernel_input_learnable else None,
                               'lr_alpha': lr_alpha, 'lbda_reg': lbda}}

    with open(PARAM_PATH, 'w') as param_file:
        json.dump(PARAM_DICT, param_file)
# ----------------------------------
# Predict and visualize
# -----------------------------------

pred_test1 = itl_estimator.model.forward(x_test, sampler_.sample(m))
pred_test2 = itl_estimator.model.forward(x_test, torch.tensor([[0.866, 0.5]], dtype=torch.float))

if plot_fig:
    plt_x = x_test[0].numpy().reshape(68, 2)
    plt_xt = pred_test1[0, 3].detach().numpy().reshape(68, 2)
    plt_uv = plt_xt - plt_x
    plt.quiver(plt_x[:, 0], plt_x[:, 1], plt_uv[:, 0], plt_uv[:, 1], angles='xy')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()
print("done")