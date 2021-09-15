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

from torch_itl.datasets import add_local_outliers
from torch_itl.estimator import FOR, SparseFOR
from torch_itl.kernel import LaplacianSum, Gaussian
from torch_itl.sampler import LinearSampler
from torch_itl.model import DecomposableIdentityScalar

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
X = add_local_outliers(X_lip, freq_loc=0.2, std=0)
Y = add_local_outliers(Y_lip, freq_loc=0.2, std=0)
n, d = X.shape
# %%
colors = [cm.viridis(x) for x in torch.linspace(0, 1, n_lip)]
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
gamma_output = torch.linspace(5, 50, 4)[3]
lbda = torch.logspace(-4, 0, 10)[6]
kernel_input = LaplacianSum(gamma_input)
kernel_output = Gaussian(gamma_output)
model = DecomposableIdentityScalar(kernel_input, kernel_output)
sampler = LinearSampler(0, 1)
est = SparseFOR(model, lbda, sampler, norm='inf')
# %%
n_eps = 40
epsilon = 0.1
intensity_list = [0.4, 0.8, 1.2]
lbda_grid = torch.logspace(-3, -1, 4)
eps_list = torch.linspace(0, 1, n_eps)
# %%
def mask(j, n):
    res = torch.ones(n, dtype=torch.bool)
    res[j] = False
    return(res)
# %%
est.epsilon = 0
est.lbda = 2e-4
est.fit_gd(X_lip, Y_lip, t, n_epoch=4000)
est.risk(X_lip, Y_lip, t)
est.plot_losses()
# %%
pred = model.forward(X_lip, t)
plt.figure()
for i in range(n_lip):
    plt.plot(t, Y_lip[i] - pred[i], color=('r'))
    plt.plot(t, pred[i])
    plt.plot(t, Y_lip[i])
plt.show()
# %%
est.get_stepsize(0)
est.dual_grad()
torch.svd(model.G_t)
model.G_t[0]
# %%
res = torch.zeros(n_eps, n_lip, len(intensity_list))
constraints = torch.zeros(n_eps, n_lip, len(intensity_list))

for k, intensity in enumerate(intensity_list):
    X = add_local_outliers(X_lip, std=intensity)
    Y = add_local_outliers(Y_lip, std=intensity)
    est.epsilon = 0
    print('Tuning lbda parameter')
    lbda, _, _ = est.tune_lambda(lbda_grid, X, Y, t, 32, fit='gd')
    est.lbda = lbda
    for j in range(n_lip):
        train_index = mask(j, n)
        test_index = ~mask(j, n)
        for i, epsilon in enumerate(eps_list):
            print('Outliers intensity:', k, 'n_split:', j, 'eps:', i)
            est.epsilon = epsilon
            est.fit_gd(X[train_index], Y[train_index], t, tol=1e-4,
                       n_epoch=4000, warm_start=True)
            res[i, j, k] = est.risk(X_lip[test_index],
                                    Y_lip[test_index], t)
            constraints[i, j, k] = est.saturated_constraints()

# %%
loo_err = res.mean(1)
plt.figure(figsize=(8, 6))
plt.title('LOO generalization error as a function of $\epsilon$', fontsize=14)
plt.xlabel('$\epsilon$', fontsize=14)
plt.tick_params(labelsize=14)
plt.ylabel('LOO generalization error', fontsize=14)
colors = [cm.viridis(x) for x in torch.linspace(0, 1, len(intensity_list))]
for i in range(len(intensity_list)):
    plt.plot(eps_list,
             loo_err[:, i], color=colors[i],
             label='$s_o$='+str(intensity_list[i]))
plt.legend(fontsize=12, loc='center right')
plt.savefig('eps_real_inf')
# %%
plt.figure()
for i in range(len(intensity_list)):
    plt.plot(eps_list, constraints.sum(1)[:, i]*0.01, color=colors[i])
plt.show()
# %%


res.mean(1)[:, 0]

res.mean(1)[:, 1]
est_eps.plot_losses()








# %%
