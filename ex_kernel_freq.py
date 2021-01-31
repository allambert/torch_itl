import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_itl import model, sampler, cost, kernel, estimator

#%%
# ----------------------------------
# Reading input/output data
# ----------------------------------
dataset = 'KDEF'  # KDEF or Rafd
theta_type = 'aff'  # aff or ''
inp_emotion = 'NE'
inc_emotion = True  # bool to include (0,0) in emotion embedding
use_facealigner = True  # bool to use aligned faces (for 'Rafd' - set to true)

data_path = os.path.join('./datasets', dataset+'_Aligned', dataset +'_LANDMARKS')  # set data path
data_emb_path = os.path.join('./datasets', dataset+'_Aligned', dataset +'_facenet')  # set data path

def get_data(dataset, kfold=0):
    if dataset == 'Rafd':
        # dirty hack only used to get Rafd speaker ids, not continuously ordered
        data_csv_path = '/home/mlpboon/Downloads/Rafd/Rafd.csv'
    affect_net_csv_path = ''  # to be set if theta_type == 'aff'
    output_folder = './LS_Experiments/'  # store all experiments in this output folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    #print('Reading data')
    if use_facealigner:
        input_data_version = 'facealigner'
        if dataset == 'KDEF':
            from datasets.datasets import kdef_landmarks_facealigner, kdef_landmarks_facenet
            x_train, y_train, x_test, y_test, train_list, test_list = \
                kdef_landmarks_facealigner(data_path, inp_emotion=inp_emotion,
                                           inc_emotion=inc_emotion, kfold=kfold)
        elif dataset == 'Rafd':
            from datasets.datasets import rafd_landmarks_facealigner
            x_train, y_train, x_test, y_test, train_list, test_list = \
                rafd_landmarks_facealigner(data_path, data_csv_path, inp_emotion=inp_emotion,
                                           inc_emotion=inc_emotion, kfold=kfold)
    else:
        from datasets.datasets import import_kdef_landmark_synthesis
        input_data_version = 'aligned2'
        x_train, y_train, x_test, y_test = import_kdef_landmark_synthesis(dtype=input_data_version)
    return (y_train, y_test)


#test of import
data_train, data_test = get_data(dataset, 1)
data_test
n,m,nf = data_train.shape
print('data dimensions', n, m, nf)
#%%
# ----------------------------------
# Set kernel and other params
# ----------------------------------

kernel_input_learnable = False
output_var_dependence = False
save_model = True
plot_fig = True
save_pred = True
get_addon_metrics = True

if kernel_input_learnable:
    NE = 10  # num epochs overall
    ne_fa = 5  # num epochs fit alpha per NE
    ne_fki = 5  # num epochs fit kernel per NE

    print('defining network')
    # define input network and kernel
    n_h = 64
    d_out = 32
    model_kernel_input = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], n_h),
        torch.nn.ReLU(),
        torch.nn.Linear(n_h, d_out)
    )

    gamma_inp = 0.1
    optim_params = dict(lr=0.001, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)

    kernel_input = kernel.LearnableGaussian(
        gamma_inp, model_kernel_input, optim_params)
else:
    NE = 1
    ne_fa = 50
    gamma_inp = 0.05
    kernel_input = kernel.Gaussian(gamma_inp)

# define emotion kernel
gamma_out = 0.39
kernel_output = kernel.Gaussian(gamma_out)

# Define PSD matrix on output variables
if output_var_dependence:
    kernel_freq = np.load('output_kernel_matrix_kdef2.npy')
else:
    kernel_freq = np.eye(nf)

# learning rate of alpha
lr_alpha = 0.01

itl_model = model.JointLandmarksSynthesisKernelModel(kernel_input, kernel_output,
                                             kernel_freq=torch.from_numpy(kernel_freq).float())

# define cost function
cost_function = cost.squared_norm_w_mask
lbda = 1e-5

affect_net_csv_path = ''
# define emotion sampler
if theta_type == 'aff':
    if dataset == 'KDEF':
        aff_emo_match = {'NE': 'Neutral',
                         'HA': 'Happy',
                         'SA': 'Sad',
                         'SU': 'Surprise',
                         'AF': 'Fear',
                         'DI': 'Disgust',
                         'AN': 'Anger',
                         }
    elif dataset == 'Rafd':
        aff_emo_match = {'neutral': 'Neutral',
                         'happy': 'Happy',
                         'sad': 'Sad',
                         'surprised': 'Surprise',
                         'fearful': 'Fear',
                         'disgusted': 'Disgust',
                         'angry': 'Anger',
                         'contemptous': 'Contempt'
                         }
    from datasets.datasets import import_affectnet_va_embedding
    aff_emo_dict = import_affectnet_va_embedding(affect_net_csv_path)
    print(aff_emo_dict)
    sampler_ = sampler.CircularSampler(data=dataset+theta_type,
                                       inp_emotion=aff_emo_match[inp_emotion],
                                       inc_emotion=inc_emotion,
                                       sample_dict=aff_emo_dict)
elif theta_type == '':
    sampler_ = sampler.CircularSampler(data=dataset,
                                       inc_emotion=inc_emotion)
sampler_.m = m
sampler_.sample(m)
#%%
itl_ridge = estimator.ITLEstimatorJoint(itl_model, cost_function, lbda, 0, sampler_)
#%%
itl_ridge.fit_closed_form(data_train)
itl_ridge.risk(data_test)
#%%
losses = torch.zeros(10,nf)
for kfold in range(10):
    data_train, data_test = get_data(dataset, kfold)
    emp_cov = data_train.reshape(-1,nf).T @ data_train.reshape(-1,nf)
    u, d, v = torch.svd(emp_cov)
    for s in range(1, nf+1):
        dim_red_model = model.LandmarksSynthesisDimReductionKernelModel(kernel_input, kernel_output, s, v, nf)
        est = estimator.ITLEstimatorJoint(dim_red_model, cost_function, lbda, 0, sampler_)
        est.fit_dim_red_closed_form(data_train)
        losses[kfold, s-1] = est.risk(data_test)

#%%
print('done computing')
d_t_r = data_train.reshape(-1, nf)
data_mean = d_t_r.mean(0)
#Y = 1/(n*m -1)*(d_t_r - data_mean.expand_as(d_t_r)).T @ (d_t_r - data_mean.expand_as(d_t_r))
Y = d_t_r.T @ d_t_r
u, d, v = torch.svd(Y)
u @ u.T
v @ v.T
torch.norm(u.T @ torch.diag_embed(d) @ u - Y)
#%%
plt.figure()
plt.plot(torch.log(d))
plt.show()
#%%
s = 60
def identity(s):
    return torch.diag_embed(torch.Tensor([1 for i in range(s)]+[0 for i in range(nf-s)]))

def proj(s):
    return torch.diag_embed(torch.Tensor([1 for i in range(s)]+[0 for i in range(nf-s)]))[:,:s]

proj(s).shape
tmp = v @ identity(s) @ v.T
#%%
s = nf-2
dim_red_model = model.LandmarksSynthesisDimReductionKernelModel(kernel_input, kernel_output, s, v, nf)
est = estimator.ITLEstimatorJoint(dim_red_model, cost_function, lbda, 0, sampler_)
est.fit_dim_red_closed_form(data_train)
est.risk(data_test)
est.model.alpha.shape
#%%
losses_train = []
losses_test = []
for s in range(nf):
    dim_red_model = model.LandmarksSynthesisDimReductionKernelModel(kernel_input, kernel_output, s, v, nf)
    est = estimator.ITLEstimatorJoint(dim_red_model, cost_function, lbda, 0, sampler_)
    est.fit_dim_red_closed_form(data_train)
    losses_train.append(est.risk(data_train))
    losses_test.append(est.risk(data_test))

losses_train
losses_test
#%%
import matplotlib.cm as cm
risks = torch.load('kdef_matrix.pt')
plt.figure()
plt.xlabel('Rank of A')
plt.ylabel('Test risk')
#plt.plot(torch.Tensor(losses_train).mean(0)[1:], c='r', label='train')
plt.plot(risks.mean(0), c= cm.viridis(0), label='test', marker='.')
plt.legend(loc='upper right')
plt.savefig('dim_red_kdef.png')
plt.show()
#%%
################################
# Tools for showing samples
###############################
class EdgeMap(object):
    def __init__(self, out_res, num_parts=3):
        self.out_res = out_res
        self.num_parts = num_parts
        self.groups = [
            [np.arange(0, 17, 1), 255],
            [np.arange(17, 22, 1), 255],
            [np.arange(22, 27, 1), 255],
            [np.arange(27, 31, 1), 255],
            [np.arange(31, 36, 1), 255],
            [list(np.arange(36, 42, 1)) + [36], 255],
            [list(np.arange(42, 48, 1)) + [42], 255],
            [list(np.arange(48, 60, 1)) + [48], 255],
            [list(np.arange(60, 68, 1)) + [60], 255]
        ]

    def __call__(self, shape):
        image = np.zeros((self.out_res, self.out_res, self.num_parts), dtype=np.float32)
        for g in self.groups:
            for i in range(len(g[0]) - 1):
                start = int(shape[g[0][i]][0]), int(shape[g[0][i]][1])
                end = int(shape[g[0][i + 1]][0]), int(shape[g[0][i + 1]][1])
                cv2.line(image, start, end, g[1], 1)
        return image

import cv2
EM = EdgeMap(out_res=128, num_parts=1)

#%%
s = 20
dim_red_model = model.LandmarksSynthesisDimReductionKernelModel(kernel_input, kernel_output, s, v, nf)
est = estimator.ITLEstimatorJoint(dim_red_model, cost_function, lbda, 0, sampler_)
est.fit_dim_red_closed_form(data_train)

example_samples = est.model.forward(data_test.reshape(-1, nf)[::7][:2], est.model.thetas)
for i in range(2):
    for j in range(m):
        im_em = EM(example_samples[i,j].detach().numpy().reshape(68,2)*128)
        plt.imshow(np.squeeze(im_em))
        plt.pause(0.3)
#%%
