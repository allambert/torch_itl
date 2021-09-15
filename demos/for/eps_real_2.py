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
from torch_itl.kernel import LaplacianSum, GaussianRFF
from torch_itl.sampler import LinearSampler
from torch_itl.model import DecomposableIntOp

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
X = add_local_outliers(X_lip, freq_loc=0.2, std=0.3)
Y = add_local_outliers(Y_lip)
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
gamma_output = torch.linspace(20, 80, 8)[2]
lbda = torch.logspace(-4, 0, 10)[6]
kernel_input = LaplacianSum(gamma_input)
kernel_output = GaussianRFF(1, 25, gamma_output)
model = DecomposableIntOp(kernel_input, kernel_output, 50)
sampler = LinearSampler(0, 1)
est = FOR(model, lbda, sampler)
# %%
n_eps = 40
epsilon = 0.1
model_eps = DecomposableIntOp(kernel_input, kernel_output, 50)
est_eps = SparseFOR(model_eps, lbda, sampler, epsilon=epsilon)
intensity_list = [0.4, 1.2, 1.7]
lbda_grid = torch.logspace(-2, 0, 5)
eps_list = torch.linspace(0, 2, n_eps)
# %%
# est_eps.epsilon=0
# est_eps.fit_gd(X_lip, Y_lip, t, n_epoch=10000, tol=1e-8)
# # %%
# est_eps.lbda = 1e-3
# est_eps.training_risk()
# pred = est_eps.predict(X_lip, t)
# plt.figure()
# for i in range(32):
#     plt.plot(t, Y_lip[i] - pred[i], color=('r'))
#     plt.plot(t, pred[i])
# plt.show()
# est_eps.plot_losses()


# %%
def mask(j, n):
    res = torch.ones(n, dtype=torch.bool)
    res[j] = False
    return(res)
# %%
res = torch.zeros(n_eps, n_lip, len(intensity_list))
constraints = torch.zeros(n_eps, n_lip, len(intensity_list))

for k, intensity in enumerate(intensity_list):
    X = add_local_outliers(X_lip, std=intensity)
    Y = add_local_outliers(Y_lip, std=intensity)
    est.epsilon = 0
    print('Tuning lbda parameter')
    lbda, _, _ = est.tune_lambda(lbda_grid, X, Y, t, 32, fit='gd')
    for j in range(n_lip):
        model_eps = DecomposableIntOp(kernel_input, kernel_output, 50)
        est_eps = SparseFOR(model_eps, lbda, sampler)
        train_index = mask(j, n)
        test_index = ~mask(j, n)
        for i, epsilon in enumerate(eps_list):
            print('Outliers intensity:', k, 'n_split:', j, 'eps:', i)
            est_eps.epsilon = epsilon
            est_eps.fit_gd(X[train_index], Y[train_index], t, tol=1e-4,
                           n_epoch=4000, warm_start=True)
            res[i, j, k] = est_eps.risk(X_lip[test_index],
                                        Y_lip[test_index], t)
            constraints[i, j, k] = est_eps.saturated_constraints()

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
plt.legend(fontsize=12, loc='upper left')
plt.savefig('eps_real_2.png', dpi=300)
# %%
plt.figure(figsize=(8, 6))
plt.title('Proportion of zero coefficients as a function of $\epsilon$',
          fontsize=14)
plt.xlabel('$\epsilon$', fontsize=14)
plt.tick_params(labelsize=14)
plt.ylabel('proportion of zero coefficients', fontsize=14)
for i in range(len(intensity_list)):
    plt.plot(eps_list,
             constraints.mean(1)[:, i], color=colors[i],
             label='$s_o$='+str(intensity_list[i]))
plt.legend(fontsize=12, loc='upper left')
plt.savefig('eps_real_2_percentage.png', dpi=300)
# %%
est_eps.model.alpha.sum()
est_eps.saturated_constraints()


constraints.mean(1)




# %%
