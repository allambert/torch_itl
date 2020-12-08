import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_itl import model, sampler, cost, kernel, estimator

# ----------------------------------
# Reading input/output data
# ----------------------------------
dataset = 'KDEF'  # KDEF or Rafd
theta_type = 'aff'  # aff or ''
inc_neutral = True  # bool to include (0,0) in emotion embedding
use_facealigner = True  # bool to use aligned faces (for 'Rafd' - set to true)

data_path = './datasets/KDEF_Aligned/KDEF_LANDMARKS'  # set data path
if dataset == 'Rafd':
    # dirty hack only used to get Rafd speaker ids, not continuously ordered
    data_csv_path = '/home/mlpboon/Downloads/Rafd/Rafd.csv'
affect_net_csv_path = './utils/landmark_utils/validation.csv' # to be set if theta_type == 'aff'
output_folder = './LS_Experiments/'  # store all experiments in this output folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

print('Reading data')
if use_facealigner:
    input_data_version = 'facealigner'
    if dataset == 'KDEF':
        from datasets.datasets import kdef_landmarks_facealigner
        x_train, y_train, x_test, y_test, train_list, test_list = \
            kdef_landmarks_facealigner(data_path, inc_neutral=inc_neutral)
    elif dataset == 'Rafd':
        from datasets.datasets import rafd_landmarks_facealigner
        x_train, y_train, x_test, y_test, train_list, test_list = \
            rafd_landmarks_facealigner(data_path, data_csv_path, inc_neutral=inc_neutral)
else:
    from datasets.datasets import import_kdef_landmark_synthesis
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
save_pred = True

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

#%%
# ----------------------------------
# Define model
# ----------------------------------
itl_model = model.SpeechSynthesisKernelModel(kernel_input, kernel_output,
                                             kernel_freq=torch.from_numpy(kernel_freq).float())

# define cost function
cost_function = cost.speech_synth_loss
lbda = 0.001

# define emotion sampler
if theta_type == 'aff':
    from datasets.datasets import import_affectnet_va_embedding
    aff_emo_dict = import_affectnet_va_embedding(affect_net_csv_path)

    sampler_ = sampler.CircularSampler(data=dataset+theta_type,
                                       inc_neutral=inc_neutral,
                                       sample_dict=aff_emo_dict)
elif theta_type == '':
    sampler_ = sampler.CircularSampler(data=dataset,
                                       inc_neutral=inc_neutral)
sampler_.m = m

itl_estimator = estimator.ITLEstimator(itl_model,
                                       cost_function, lbda, sampler_)


#%%
# ----------------------------------
# Training
# ----------------------------------

for ne in range(NE):
    itl_estimator.fit_alpha(x_train, y_train, n_epochs=ne_fa,
                        lr=lr_alpha, line_search_fn='strong_wolfe', warm_start=False)

    print(itl_estimator.losses)
    #itl_estimator.clear_memory()
    if kernel_input_learnable:
        itl_estimator.fit_kernel_input(x_train, y_train, n_epochs=ne_fki)
        print(itl_estimator.model.kernel_input.losses)
        itl_estimator.model.kernel_input.clear_memory()

#%%
itl_estimator.fit_closed_form(x_train, y_train)
itl_estimator.training_risk()
#%%
# ----------------------------------
# Save model and params
# -----------------------------------

if save_model:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    MODEL_DIR = os.path.join(output_folder, dataset + '_itl_model_' + timestr, 'model')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    PATH =  os.path.join(MODEL_DIR, 'itl_model_ckpt_' + timestr + '.pt')
    torch.save({'itl_alpha': itl_estimator.model.alpha.data,
                'itl_kernel_input': model_kernel_input.state_dict() if kernel_input_learnable else None,
                'itl_kernel_output': None}, PATH)

    PARAM_PATH = os.path.join(MODEL_DIR, 'itl_model_config_' + timestr + '.json')
    PARAM_DICT = {'Data': {'n': n, 'm': m, 'nf': nf,
                           'input_data_version': input_data_version,
                           'dataset': dataset,
                           'theta_type': theta_type,
                           'include_neutral': inc_neutral},
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

#%%
# ----------------------------------
# Predict and visualize
# -----------------------------------

pred_train = itl_estimator.model.forward(x_train, sampler_.sample(m))
pred_test1 = itl_estimator.model.forward(x_test, sampler_.sample(m))
pred_test2 = itl_estimator.model.forward(x_test, torch.tensor([[0.866, 0.5]], dtype=torch.float))

if use_facealigner and save_pred:
    PRED_DIR = os.path.join(output_folder, dataset + '_itl_model_' + timestr, 'predictions', dataset)
    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)
    pred_test1_np = pred_test1.detach().numpy()
    # x_test_np = np.repeat(x_test.numpy()[:, np.newaxis, :], 6, axis=1)
    pred_test1_np = pred_test1_np*128
    for i in range(pred_test1_np.shape[0]):
        for j in range(pred_test1_np.shape[1]):
            np.savetxt(os.path.join(PRED_DIR, 'pred_' + test_list[i][1][j]),
                       pred_test1_np[i, j].reshape(68, 2))

if plot_fig:
    plt_x = x_test[0].numpy().reshape(68, 2)
    plt_xt = pred_test1[0, 5].detach().numpy().reshape(68, 2)
    if use_facealigner:
        plt_x = plt_x * 128
        plt_xt = plt_xt * 128
    plt_uv = plt_xt - plt_x
    plt.quiver(plt_x[:, 0], plt_x[:, 1], plt_uv[:, 0], plt_uv[:, 1], angles='xy')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()
print("done")

#%%
# ----------------------------------
# Provide Risk
# -----------------------------------
plt.figure()
plt.plot(itl_estimator.losses)
plt.show()
print("Empirical Risk Train:", itl_estimator.cost(y_train, pred_train, sampler_.sample(m)))
print("Empirical Risk Test:", itl_estimator.cost(y_test, pred_test1, sampler_.sample(m)))
print("Estimator Norm:", itl_estimator.model.vv_norm())
