import os
import sys
import importlib

if importlib.util.find_spec('torch_itl') is None:
    path_to_lib = os.getcwd()[:-15]
    sys.path.append(path_to_lib)

from torch_itl.estimator import IQR
from torch_itl.kernel import Gaussian, LearnableGaussian
from torch_itl.model import DecomposableIdentity
from torch_itl.sampler import LinearSampler
from torch_itl.datasets import import_data_toy_quantile
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %%
# Defining a simple toy dataset:

print("Creating the dataset")

x_train, y_train, _ = import_data_toy_quantile(150)
n = x_train.shape[0]
m = 10

plt.figure()
plt.scatter(x_train, y_train, marker='.')
plt.show()

# %%
# Defining an ITL model, first without a learnable kernel

print("Defining the model")

kernel_input = Gaussian(3.5)
kernel_output = Gaussian(9)
model = DecomposableIdentity(kernel_input, kernel_output, 1)
lbda = 0.001
lbda_cross = 0.01
sampler = LinearSampler(0.1, 0.9, 10, 0)
sampler.m = 10
est = IQR(model, lbda, lbda_cross, sampler)
#%%
# Learning the coefficients of the model
print("Fitting the coefficients of the model")

est.fit_alpha_gd(x_train, y_train, n_epochs=40,
                        lr=0.001, line_search_fn='strong_wolfe')
#%%
# Plotting the loss along learning

plt.figure()
plt.title("Loss evolution with time")
plt.plot(est.losses)
plt.show()
best_loss = est.losses[-1]

# Plotting the model on test points

probs = est.sampler.sample(30)
x_test = torch.linspace(0, 1.4, 100).view(-1, 1)
y_pred = est.model.forward(x_test, probs).detach().numpy()
colors = [cm.viridis(x.item()) for x in torch.linspace(0, 1, 30)]
plt.figure()
plt.title("Conditional Quantiles output by our model")
plt.scatter(x_train, y_train, marker='.')
for i in range(30):
    plt.plot(x_test, y_pred[:, i], c=colors[i])
plt.show()

# %%
# Let's learn the input kernel with ITL
# First define a neural net

n_h = 40
d_out = 10
model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(x_train.shape[1], n_h),
    torch.nn.ReLU(),
    torch.nn.Linear(n_h, n_h),
    torch.nn.Linear(n_h, d_out),
)
gamma = 3
optim_params = dict(lr=0.01, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)

kernel_input = LearnableGaussian(gamma, model_kernel_input, optim_params)
est.model.kernel_input = kernel_input

# %%

est.fit_kernel_input(x_train, y_train)

# plot the loss along learning the kernel
#%%
plt.figure()
plt.title("Loss evolution when learning the kernel")
plt.plot(est.model.kernel_input.losses)
plt.show()

# %%
# Now retrain the parameters alpha of the model
est.fit_alpha_gd(x_train, y_train, n_epochs=40,
                        lr=0.01, line_search_fn='strong_wolfe')

# plot the loss

plt.figure()
plt.title("Loss evolution when learning model coefficients again")
plt.plot(est.losses)
plt.show()

y_pred = est.model.forward(x_test, probs).detach().numpy()
colors = [cm.viridis(x.item()) for x in torch.linspace(0, 1, 30)]
plt.figure()
plt.title('Conditional Quantiles with learned kernel')
plt.scatter(x_train, y_train, marker='.')
for i in range(30):
    plt.plot(x_test, y_pred[:, i], c=colors[i])
plt.show()

print('Loss gain from learning the kernel: ',
      best_loss - est.losses[-1])
