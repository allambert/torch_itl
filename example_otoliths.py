import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch_itl import model, sampler, cost, kernel, estimator
from datasets.datasets import import_data_otoliths

# %%
# Data import

X_train, Y_train, X_test, Y_test = import_data_otoliths()

# %%
# Conditional Quantiles computation

dtype = torch.float
device = torch.device("cpu")

n_h = 80
d_out = 50
model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], n_h),
    torch.nn.ReLU(),
    torch.nn.Linear(n_h, n_h),
    torch.nn.Linear(n_h, d_out),
)
optim_params = dict(lr=0.001, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)

kernel_input = kernel.GaussianRFF(X_train.shape[1],50,3)
kernel_output = kernel.GaussianRFF(1,30,4)
itl_model = model.RFFModel(kernel_input, kernel_output)
cost_function = cost.ploss_with_crossing(0.01)
lbda = 0.001
sampler_ = sampler.LinearSampler(0.05, 0.95, epsilon=0)
sampler_.m = 25

itl_estimator = estimator.ITLEstimator(itl_model,
                                       cost_function, lbda, sampler_)

itl_estimator.fit_alpha(X_train, Y_train, n_epochs=20,
                        lr=0.001, line_search_fn='strong_wolfe')

# %%

plt.figure()
plt.plot(itl_estimator.losses)
plt.show()

# %%
