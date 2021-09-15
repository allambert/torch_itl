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

from torch_itl.datasets import SyntheticGPmixture, add_local_outliers
from torch_itl.estimator import FOR, SparseFOR
from torch_itl.kernel import LaplacianSum, Gaussian
from torch_itl.sampler import LinearSampler
from torch_itl.model import DecomposableIdentityScalar

# %%
n_i = 250
n_o = 50
n_atoms = 4
gamma_cov_i = torch.Tensor([[0.05, 0.1, 0.5, 0.7],
                            [0.05, 0.1, 0.5, 0.7]]).numpy()
gamma_cov_o = torch.Tensor([[0.01, 0.1, 1, 4],
                            [0.01, 0.1, 1, 4]]).numpy()

noise = [None, None]
scale = 1.5
data_inliers = SyntheticGPmixture(n_atoms=n_atoms, gamma_cov=gamma_cov_i,
                                  noise=noise, scale=scale)
data_outliers = SyntheticGPmixture(n_atoms=n_atoms, gamma_cov=gamma_cov_o,
                                   noise=noise, scale=2.5)
# %%
t = torch.linspace(0, 1, 70)
# Generating inliers
data_inliers.drawGP(t)
X_i, Y_i = data_inliers.sample(n_i)
colors = [cm.viridis(x) for x in torch.linspace(0, 1, n_i)]
# Generating outliers
X = add_local_outliers(X_i, std=0)
Y = add_local_outliers(Y_i, std=0)
plt.figure()
plt.subplot(211)
plt.title('Input inliers functions $(x_i)_{i=1}^n$')
for i in range(n_i):
    plt.plot(t, X[i], color=colors[i])
plt.subplot(212)
plt.tight_layout()
plt.title('Output inliers functions $(y_i)_{i=1}^n$')
for i in range(n_i):
    plt.plot(t, Y[i], color=colors[i])
plt.show(block=None)
# %%
kernel_input = LaplacianSum(5)
kernel_output = Gaussian(40)
model = DecomposableIdentityScalar(kernel_input, kernel_output)
sampler = LinearSampler(0, 1)
lbda_grid = torch.logspace(-6, -2, 8)
model_eps = DecomposableIdentityScalar(kernel_input, kernel_output)
est_eps = SparseFOR(model_eps, 2e-5, sampler, norm='inf')
est_eps.epsilon = 0.1
# lbda = est_eps.tune_lambda(lbda_grid, X, Y, t.view(-1, 1), n_splits=5, fit='gd')
# est_eps.lbda = lbda[0]
# est.plot_losses()
# %%
est_eps.fit_cd(X, Y, t.view(-1, 1), n_epoch=100, warm_start=True)
est_eps.plot_losses()

# %%
n_test = 30
X_test, Y_test = data_inliers.sample(n_test)
pred = model_eps.forward(X_test, t.view(-1, 1))
print('MSE:', torch.mean((Y_test - pred)**2).detach().item())
print('MSE predictor 0:', torch.mean(Y_test**2).item())
plt.figure()
plt.title("Plot of the residuals")
colors = [cm.viridis(x) for x in torch.linspace(0, 1, n_test)]
for i in range(n_test):
    plt.plot(t, pred[i], color=colors[i])
    plt.plot(t, Y_test[i], color=colors[i])
    plt.plot(t, pred[i] - Y_test[i], color='r')
plt.show(block=None)
# %%
est_eps.saturated_constraints()
n_epsilon = 40
epsilon_list = torch.linspace(0, 1, n_epsilon)
# %%
intensity_list = [0.2, 0.5, 0.8, 1.2]
res = torch.zeros(n_epsilon, len(intensity_list))
constraints = torch.zeros(n_epsilon, len(intensity_list))
for j in range(len(intensity_list)):
    X = add_local_outliers(X_i, std=intensity_list[j])
    Y = add_local_outliers(Y_i, std=intensity_list[j])
    print('Tuning lbda parameter')
    est_eps.epsilon = 0
    lbda, _, _ = est_eps.tune_lambda(lbda_grid, X, Y, t.view(-1, 1),
                                     n_splits=5)
    est_eps.lbda = lbda
    for i, epsilon in enumerate(epsilon_list):
        est_eps.epsilon = epsilon
        est_eps.fit_gd(X, Y, t.view(-1, 1), n_epoch=2500, warm_start=True)
        res[i, j] = est_eps.risk(X_test, Y_test, t.view(-1, 1))
        constraints[i, j] = est_eps.saturated_constraints()
        print('Done with intensity:', j, 'epsilon:', i)
# %%
torch.save(res, 'eps_synt_inf')
# %%
plt.figure(figsize=(8, 6))
plt.title('Test MSE as a function of $\epsilon$', fontsize=14)
plt.xlabel('$\epsilon$', fontsize=14)
plt.xlim(-0.03, 1.5)
plt.tick_params(labelsize=14)
plt.ylabel('Test MSE', fontsize=14)
colors = [cm.viridis(x) for x in torch.linspace(0, 1, len(intensity_list))]
plt.hlines(0.1, 0, 0, colors='grey',
           linestyles='--', label='Ridge Regression')
for i in range(len(intensity_list)):
    plt.plot(torch.linspace(0, epsilon_max_list[i], n_epsilon),
             res[:, i].flip(0), color=colors[i],
             label='$s_o$='+str(intensity_list[i]))
    plt.hlines(res[0, i], epsilon_max_list[i], 4.5,
               color=colors[i], linestyles='--')
plt.legend(fontsize=12, loc='center right')
plt.savefig('eps_synt_inf')
# %%
