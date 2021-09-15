import os
import sys
import importlib
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

if importlib.util.find_spec('torch_itl') is None:
    path_to_lib = os.getcwd()[:-10]
    sys.path.append(path_to_lib)

from torch_itl.datasets import add_global_outliers_worse
from torch_itl.estimator import FOR, RobustFOR
from torch_itl.kernel import GaussianSum, Harmonic, LaplacianSum, GaussianRFF
from torch_itl.sampler import LinearSampler
from torch_itl.model import DecomposableIdentity, DecomposableIntOp

# %%

print('Importing data...')
pfx = '/Users/alambert/Recherche/ITL/code/torch_itl/torch_itl/datasets/FOR/Lip'
data_filename = '/EMGmatlag.csv'
target_filename = '/lipmatlag.csv'
sampling_filename = '/tfine.csv'

data = pd.read_csv(pfx + data_filename, header=None)
target = pd.read_csv(pfx + target_filename, header=None)
sampling = pd.read_csv(pfx + sampling_filename, header=None)

# %%
X_lip = torch.from_numpy(data.values.T).float()
Y_lip = torch.from_numpy(target.values.T).float()
t = torch.from_numpy(sampling.values.reshape(-1)).float().view(-1, 1)
t = t / 0.64
n_lip = X_lip.shape[0]
print('Size of X:', X_lip.shape)
print('Size of Y:', Y_lip.shape)
print('Size of t:', t.shape)
n_o = 4
X, Y = add_global_outliers_worse(X_lip, Y_lip, n_o, 1.2, seed='fixed')
n, d = X.shape
print('New size of X:', X.shape)
# %%
tmp = torch.zeros(641, 641)
for i in range(641):
    for j in range(641):
        tmp[i, j] = (t[i] - t[j])**2

1 /tmp.view(-1).sort().values[int(641*641*0.2)]
# %%
colors = [cm.viridis(x) for x in torch.linspace(0, 1, n_lip)] + ['r']*n_o
color_out = 'r'
plt.figure()
plt.subplot(211)
plt.title('Input functions $(x_i)_{i=1}^n$')
for i in range(n):
    plt.plot(t, X[i], color=colors[i])
plt.subplot(212)
plt.tight_layout()
plt.title('Output functions $(y_i)_{i=1}^n$')
for i in range(n):
    plt.plot(t, Y[i], color=colors[i])
plt.show(block=None)

# %%
gamma_input = torch.linspace(5, 50, 8)[0]
gamma_output = torch.linspace(20, 80, 8)[2]
lbda = torch.logspace(-4, 0, 10)[6]
kernel_input = LaplacianSum(gamma_input)
kernel_output = GaussianRFF(1, 25, gamma_output)
model = DecomposableIntOp(kernel_input, kernel_output, 50)
sampler = LinearSampler(0, 1)
est = FOR(model, lbda, sampler)
# %%
n_kappa = 40
kappa = 0.1
model_rob = DecomposableIntOp(kernel_input, kernel_output, 50)
est_rob = RobustFOR(model_rob, lbda, sampler, kappa=kappa)
intensity_list = [0.5, 1, 2, 5]
lbda_grid = torch.logspace(-2, 0, 5)
kappa_max_list = torch.zeros(len(intensity_list))
# %%
def mask(j, n):
    res = torch.ones(n, dtype=torch.bool)
    res[j] = False
    return(res)
# %%
res = torch.zeros(n_kappa, n_lip, len(intensity_list))
for k, intensity in enumerate(intensity_list):
    X, Y = add_global_outliers_worse(X_lip, Y_lip, n_o, intensity,
                                     seed='fixed')
    print('Tuning lbda parameter')
    lbda, _, _ = est.tune_lambda(lbda_grid, X, Y, t, 36)
    kappa_max_list[k] = est.get_kappa_max('2')
    kappa_list_2 = torch.linspace(0, kappa_max_list[k]*1.2, n_kappa)
    for j in range(n_lip):
        model_rob = DecomposableIntOp(kernel_input, kernel_output, 50)
        est_rob = RobustFOR(model_rob, lbda, sampler, kappa=kappa)
        train_index = mask(j, n)
        test_index = ~mask(j, n)
        for i, kappa in enumerate(kappa_list_2):
            print('Outliers intensity:', k, 'n_split:', j, 'kappa:', i)
            est_rob.kappa = kappa
            est_rob.fit_cd(X[train_index], Y[train_index], t, tol=1e-4,
                           n_epoch=2000, warm_start=True)
            res[i, j, k] = est_rob.risk(X[test_index], Y[test_index], t)

# %%
#torch.save(res, 'huber_real_2')
res = torch.load('huber_real_2')
# %%
loo_err = res.mean(1)
plt.figure()
plt.title('LOO generalization error as a function of $\kappa$', fontsize=14)
plt.xlabel('$\kappa$', fontsize=14)
plt.tick_params(labelsize=14)
plt.ylabel('LOO generalization error', fontsize=14)
colors = [cm.viridis(x) for x in torch.linspace(0, 1, len(intensity_list))]
plt.hlines(0.8, 0, 0, colors='grey',
           linestyles='--', label='Ridge Regression')
for i in range(len(intensity_list)):
    plt.plot(torch.linspace(0, kappa_max_list[i]*1.2, n_kappa),
             loo_err[:, i], color=colors[i],
             label='$s_o$='+str(intensity_list[i]))
    plt.hlines(loo_err[-1, i], kappa_max_list[i]*1.2, 7,
               color=colors[i], linestyles='--')
plt.legend(fontsize=12, loc='center right')
plt.savefig('huber_real_2.png', dpi=300)
# %%
est_rob.plot_losses()














# %%
