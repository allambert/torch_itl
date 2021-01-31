import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch_itl import model, sampler, cost, kernel, estimator

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
    gamma_inp = 0.1
    kernel_input = kernel.Gaussian(gamma_inp)

# define emotion kernel
gamma_out = 0.5
kernel_output = kernel.Gaussian(gamma_out)

# Define PSD matrix on output variables
if output_var_dependence:
    kernel_freq = np.load('output_kernel_matrix_kdef2.npy')
else:
    kernel_freq = np.eye(nf)

# learning rate of alpha
lr_alpha = 0.01
#%%
gamma_inp_list = torch.load('KDEF_joint_emotion_gamma_inp.pt')
gamma_out_list = torch.load('KDEF_joint_emotion_gamma_out.pt')
lbda_list = torch.load('KDEF_joint_emotion_lbdas.pt')
#%%
# ----------------------------------
# Define model
# ----------------------------------
itl_model = model.JointLandmarksSynthesisKernelModel(kernel_input, kernel_output,
                                             kernel_freq=torch.from_numpy(kernel_freq).float())

# define cost function
cost_function = cost.squared_norm_w_mask
lbda = 0.0001

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
mask = torch.ones(n,m,dtype=torch.bool)
itl_estimator = estimator.ITLEstimatorJointPartial(itl_model, cost_function, lbda, 0, sampler_, mask)

#%%
# ----------------------------------
# Training
# ----------------------------------
n_loops = 4
#results tensor
test_losses = torch.zeros(10, n_loops, n)

for kfold in range(10):
    data_train, data_test = get_data(dataset, kfold)
    mask_list = [torch.randperm(n*m).reshape(n, m) for j in range(n_loops)]
    for j in range(n_loops):
        mask_level = mask_list[j]
        for i in torch.arange(n*m)[::7]:
            itl_estimator.mask = (mask_level >= i)
            itl_estimator.fit_closed_form(data_train)
            test_losses[kfold,j,i//7] = itl_estimator.risk(data_test)
    print('done with kfold ', kfold)

#%%
torch.save(test_losses, 'kdef_partial.pt')
#%%
risks_kdef = torch.load('kdef_partial.pt').mean(0).mean(0)
risks_rafd = torch.load('rafd_partial.pt').mean(0).mean(0)
idx_loss_kdef = torch.arange(n*m)[::7].float() / n / m
idx_loss_rafd = torch.arange(risks_rafd.shape[0]*m)[::7].float() / risks_rafd.shape[0]/m
min_risk_kdef = risks_kdef[0]
min_risk_rafd = risks_rafd[0]
log_risks_kdef = torch.log10(risks_kdef)
log_risks_rafd = torch.log10(risks_rafd)
#%%
plt.figure()
plt.xlabel("% of missing data")
plt.ylabel("$\log_{10}$ Test MSE")
colors = [cm.viridis(i) for i in torch.linspace(0,0.8,2)]
plt.plot(idx_loss_kdef, log_risks_kdef, c='black', label='KDEF', marker=',')
plt.plot(idx_loss_rafd, log_risks_rafd, c='grey', label='RaFD', marker=',')
plt.legend(loc='upper left')
plt.savefig('partial_observation.pdf')
plt.show()
#%%
torch.exp(torch.Tensor([0.5]))
itl_estimator.fit_closed_form(y_train)
#
# itl_estimator.model.G_x[1::7][:,1::7]
#
 itl_estimator.risk(y_train, mask)
itl_estimator.risk(y_test)
# itl_estimator.lbda_reg = 0.0000001
# itl_estimator.mask = mask
# itl_estimator.model.alpha[::7]
 itl_estimator.model.kernel_input.gamma
#%%










#%%
