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
inp_emotion = 'NE'
inc_emotion = True  # bool to include (0,0) in emotion embedding
use_facealigner = True  # bool to use aligned faces (for 'Rafd' - set to true)

data_path = os.path.join('./datasets', dataset+'_Aligned', dataset +'_LANDMARKS')  # set data path
data_emb_path = os.path.join('./datasets', dataset+'_Aligned', dataset +'_facenet')  # set data path

if dataset == 'Rafd':
    # dirty hack only used to get Rafd speaker ids, not continuously ordered
    data_csv_path = '/home/mlpboon/Downloads/Rafd/Rafd.csv'
affect_net_csv_path = ''  # to be set if theta_type == 'aff'
output_folder = './LS_Experiments/'  # store all experiments in this output folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

print('Reading data')
if use_facealigner:
    input_data_version = 'facealigner'
    if dataset == 'KDEF':
        from datasets.datasets import kdef_landmarks_facealigner, kdef_landmarks_facenet
        # x_train, y_train, x_test, y_test, train_list, test_list = \
        #     kdef_landmarks_facealigner(data_path, inp_emotion=inp_emotion,
        #                                inc_emotion=inc_emotion)
        x_train, y_train, x_test, y_test, train_list, test_list = \
            kdef_landmarks_facenet(data_path, data_emb_path, inp_emotion=inp_emotion,
                                       inc_emotion=inc_emotion)
    elif dataset == 'Rafd':
        from datasets.datasets import rafd_landmarks_facealigner
        x_train, y_train, x_test, y_test, train_list, test_list = \
            rafd_landmarks_facealigner(data_path, data_csv_path, inp_emotion=inp_emotion,
                                       inc_emotion=inc_emotion)
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

    gamma_inp = 3
    optim_params = dict(lr=0.001, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)

    kernel_input = kernel.LearnableGaussian(
        gamma_inp, model_kernel_input, optim_params)
else:
    NE = 1
    ne_fa = 50
    gamma_inp = 0.2
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

itl_model = model.JointLandmarksSynthesisKernelModel(kernel_input, kernel_output,
                                             kernel_freq=torch.from_numpy(kernel_freq).float())

# define cost function
cost_function = cost.squared_norm_w_mask
lbda = 0.001/7

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
mask = torch.ones(n,m,dtype=torch.bool)
#%%
itl_estimator = estimator.ITLEstimatorJoint(itl_model, cost_function, lbda, 0, sampler_)
#%%
# ----------------------------------
# Cross validation loop
# -----------------------------------







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
                               'lr_alpha': lr_alpha, 'lbda_reg': lbda_reg}}

    with open(PARAM_PATH, 'w') as param_file:
        json.dump(PARAM_DICT, param_file)
x_train.shape
#%%
# ----------------------------------
# Predict and visualize
# -----------------------------------
y_test.shape
x_test.shape
x_test_joint = y_test.reshape(-1,136)
y_test_joint = torch.zeros(14*7, 7, 136)
for i in range(14*7):
    y_test_joint[i] = y_test[i//7]
y_test_joint.shape

pred_train = itl_estimator.model.forward(y_train.reshape(-1,136), sampler_.sample(m))
pred_test1 = itl_estimator.model.forward(x_test_joint, sampler_.sample(m))
pred_test2 = itl_estimator.model.forward(x_test_joint, torch.tensor([[0.866, 0.5]], dtype=torch.float))
#%%
if use_facealigner and save_pred:
    PRED_DIR = os.path.join(output_folder, dataset + '_itl_model_' + timestr, 'predictions', dataset)
    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)
    pred_test1_np = pred_test1.detach().numpy()
    # x_test_np = np.repeat(x_test.numpy()[:, np.newaxis, :], 6, axis=1)
    pred_test1_np = pred_test1_np*128
    for i in range(pred_test1_np.shape[0]):
        for j in range(pred_test1_np.shape[1]):
            np.savetxt(os.path.join(PRED_DIR, 'pred_' + test_list[i//7][1][j]),
                       pred_test1_np[i, j].reshape(68, 2))

if plot_fig:
    plt_x = x_test[0].numpy().reshape(68, 2)
    plt_xt = pred_test1[3, 6].detach().numpy().reshape(68, 2)
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
# plt.figure()
# plt.plot(itl_estimator.losses)
# plt.show()
# print("Empirical Risk Train:", itl_estimator.cost(y_train_joint, pred_train, sampler_.sample(m)))
itl_estimator.cost(y_test_joint, pred_test1, sampler_.sample(m)).mean(axis=1)
print("Empirical Risk Train:",itl_estimator.training_risk())
y_test_joint.shape
print("Empirical Risk Test:", itl_estimator.cost(y_test_joint, pred_test1, sampler_.sample(m)))
print("Estimator Norm:", itl_estimator.model.vv_norm())

#%%
# Generating edgemaps
def circular_sampling(theta1, theta2, num_samples):
    angle1 = np.arctan2(theta1[1], theta1[0])
    angle2 = np.arctan2(theta2[1], theta2[0])
    angle1 = angle1 if angle1>=0 else angle1+(2*np.pi)
    angle2 = angle2 if angle2>=0 else angle2+(2*np.pi)

    reverse = False
    if angle1>angle2:
        start = angle2; end = angle1
        reverse = True
    else:
        start = angle1; end = angle2

    sampled_angles = np.linspace(start=start, stop=end, num=num_samples, endpoint=True)
    sample_coords = np.vstack((np.cos(sampled_angles), np.sin(sampled_angles))).T

    if reverse:
        return np.flipud(sample_coords)
    else:
        return sample_coords

def radial_sampling(theta, num_samples):
    angle = np.arctan2(theta[1], theta[0])
    sampled_radii = np.linspace(start=0, stop=1, num=num_samples, endpoint=True)
    sample_coords = np.vstack((sampled_radii*np.cos(angle), sampled_radii*np.sin(angle))).T
    return sample_coords


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
sampling_type = 'radial'
num_samples = 10
if sampling_type == 'circular':
    sampled_emotions = circular_sampling(aff_emo_dict['Happy'], aff_emo_dict['Surprise'], num_samples)
elif sampling_type == 'radial':
    sampled_emotions = radial_sampling(aff_emo_dict['Happy'], num_samples)
EM = EdgeMap(out_res=128, num_parts=1)
#%%

#%matplotlib inline
for i in range(len(sampled_emotions)):
    pred_test = itl_model.forward(x_test_joint, torch.from_numpy(sampled_emotions[i][np.newaxis]).float())
    im_em = EM(pred_test[0, 0].detach().numpy().reshape(68,2)*128)
    plt.imshow(np.squeeze(im_em))
    plt.pause(0.5)

#%%

#%matplotlib inline
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
imlist = []
pred_test1.size()
for i in range(98):
    for j in range(7):
        im_em = EM(pred_test1[i, j].detach().numpy().reshape(68,2)*128)
        imlist.append(transforms.ToTensor()(im_em.copy()))
    #imlist.append(transforms.ToTensor()(im_em.copy()))
#show(make_grid(imlist, nrow=10, padding=10, pad_value=1))
#save_image(imlist, 'radial_happy_to_surprise.jpg', nrow=10, padding=10, pad_value=1)

#%%

from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if dataset == 'KDEF':
    num_classes = 7
    MODEL_PATH = './utils/KDEF_bs16_e10_20201117-181507'
elif dataset == 'Rafd':
    num_classes = 8
    MODEL_PATH = './utils/landmark_utils/Classification/LndExperiments/Rafd_bs16_e10_20201118-055249'

# model def
def model(model_name, num_classes):
    if model_name == 'resnet-18':
        model_ft = models.resnet18(pretrained=False)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


# Get ResNet and load wts
emo_model_ft = model('resnet-18', num_classes)
emo_model_ft.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
emo_model_ft = emo_model_ft.to(device)
emo_model_ft.eval()

inputs = F.interpolate(torch.stack(imlist), size=224, mode='bilinear')
outputs = emo_model_ft(inputs/255.)
sout = nn.functional.softmax(outputs, dim=1)
sout_np = sout.detach().numpy()

#%%
tmp = np.argmax(sout_np, axis=1)
tmp = tmp.reshape(-1,7)
cmp = np.zeros_like(tmp)
for i in range(4):
    cmp[:,i] = i
cmp[:,4]  = 5
cmp[:,5]  = 6
cmp[:,6]  = 4

res = (cmp == tmp)
print('Average accuracy over test set:',res.mean())
#%%
tmp
emo_idx = np.array([0,1,2,3,6,4,5])
confusion_matrix = np.zeros((7,7))
for i in range(7):
    for j in range(7):
        confusion_matrix[i,j] = np.where(tmp[:,i] == j, np.ones(98), np.zeros(98)).sum()

confusion_matrix
