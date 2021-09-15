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

from torch_itl.datasets import SyntheticGPmixture
from torch_itl.estimator import FOR, RobustFOR
from torch_itl.kernel import LaplacianSum, GaussianRFF
from torch_itl.sampler import LinearSampler
from torch_itl.model import DecomposableIntOp

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
plt.figure()
plt.subplot(211)
plt.title('Input inliers functions $(x_i)_{i=1}^n$')
for i in range(n_i):
    plt.plot(t, X_i[i], color=colors[i])
plt.subplot(212)
plt.tight_layout()
plt.title('Output inliers functions $(y_i)_{i=1}^n$')
for i in range(n_i):
    plt.plot(t, Y_i[i], color=colors[i])
plt.show(block=None)
# Generating outliers
data_outliers.drawGP(t)
X_o, Y_o = data_outliers.sample(n_o)
colors = [cm.viridis(x) for x in torch.linspace(0, 1, n_o)]
plt.figure()
plt.subplot(211)
plt.title('Input outliers functions $(x_i)_{i=1}^n$')
for i in range(n_o):
    plt.plot(t, X_o[i], color=colors[i])
plt.subplot(212)
plt.tight_layout()
plt.title('Output outliers functions $(y_i)_{i=1}^n$')
for i in range(n_o):
    plt.plot(t, Y_o[i], color=colors[i])
plt.show(block=None)
# Gathering dataset
X, Y = torch.cat((X_i, X_o), 0), torch.cat((Y_i, Y_o), 0)
# %%
kernel_input = LaplacianSum(5)
kernel_output = GaussianRFF(1, 25, 40)
model = DecomposableIntOp(kernel_input, kernel_output, 50)
lbda = 2e-5
sampler = LinearSampler(0, 1)
est = FOR(model, lbda, sampler)
lbda_grid = torch.logspace(-6, -2, 8)
# lbda = est.tune_lambda(lbda_grid, X, Y, t.view(-1, 1), n_splits=5)
# est.plot_losses()
# %%
est.fit_cd(X, Y, t.view(-1, 1), warm_start=False)
est.plot_losses()
# %%
n_test = 30
X_test, Y_test = data_inliers.sample(n_test)
pred = model.forward(X_test, t.view(-1, 1))
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
kappa_max = est.get_kappa_max('2')
n_kappa = 40
kappa_list = torch.linspace(0, kappa_max, n_kappa).flip(0)
model_rob = DecomposableIntOp(kernel_input, kernel_output, 50)
est_rob = RobustFOR(model_rob, lbda, sampler)
# %%
intensity_list = [2, 4, 6, 10]
res = torch.zeros(n_kappa, len(intensity_list))
kappa_max_list = torch.zeros(4)
for j in range(len(intensity_list)):
    data_outliers = SyntheticGPmixture(n_atoms=n_atoms, gamma_cov=gamma_cov_o,
                                       noise=noise, scale=intensity_list[j])
    data_outliers.drawGP(t)
    X_o, Y_o = data_outliers.sample(n_o)
    X, Y = torch.cat((X_i, X_o), 0), torch.cat((Y_i, Y_o), 0)
    print('Tuning lbda parameter')
    lbda, _, _ = est.tune_lambda(lbda_grid, X, Y, t.view(-1, 1), n_splits=5)
    est.lbda = lbda
    est.fit_cd(X, Y, t.view(-1, 1), warm_start=False)
    kappa_max_list[j] = est.get_kappa_max('2')
    kappa_list = torch.linspace(0, kappa_max_list[j], n_kappa).flip(0)
    est_rob.lbda = lbda
    for i, kappa in enumerate(kappa_list):
        est_rob.kappa = kappa
        est_rob.fit_cd(X, Y, t.view(-1, 1), n_epoch=2500, warm_start=True)
        res[i, j] = est_rob.risk(X_test, Y_test, t.view(-1, 1))
        print('Done with intensity:', j, 'kappa:', i)
# %%
torch.save(res, 'huber_synt_2')
# %%
plt.figure()
plt.title('Test MSE as a function of $\kappa$', fontsize=14)
plt.xlabel('$\kappa$', fontsize=14)
plt.xlim(-0.03, 1.5)
plt.tick_params(labelsize=14)
plt.ylabel('Test MSE', fontsize=14)
colors = [cm.viridis(x) for x in torch.linspace(0, 1, len(intensity_list))]
plt.hlines(0.1, 0, 0, colors='grey',
           linestyles='--', label='Ridge Regression')
for i in range(len(intensity_list)):
    plt.plot(torch.linspace(0, kappa_max_list[i], n_kappa),
             res[:, i].flip(0), color=colors[i],
             label='$s_o$='+str(intensity_list[i]))
    plt.hlines(res[0, i], kappa_max_list[i], 4.5,
               color=colors[i], linestyles='--')
plt.legend(fontsize=12, loc='center right')
plt.savefig('huber_synt_2.png', dpi=300)
# %%
kappa_max_list
