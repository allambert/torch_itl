import torch
import numpy as np
import matplotlib. pyplot as plt
import matplotlib.cm as cm

from torch_itl.datasets import SyntheticGPmixture

# Torch default datatype
torch.set_default_dtype(torch.float64)


# Generate synthetic dataset ###########################################
# ######################################################################
n_train = 100
n_test = 100
seed_train = 765
seed_test = 443
# Sampling locations
theta = torch.linspace(0, 1, 100)
# Generate data
data_synth = SyntheticGPmixture()
data_synth.drawGP(theta)
Xtrain, Ytrain = data_synth.sample(n_train, seed_coefs=seed_train)
Xtest, Ytest = data_synth.sample(n_test, seed_coefs=seed_test)

n_plot = n_train
colors = [cm.viridis(x) for x in np.linspace(0, 1, n_plot)]
plt.figure()
plt.subplot(211)
plt.title('Input functions $(x_i)_{i=1}^n$')
for i in range(n_plot):
    plt.plot(theta, Xtrain[i], color=colors[i])
plt.subplot(212)
plt.tight_layout()
plt.title('Output functions $(y_i)_{i=1}^n$')
for i in range(n_plot):
    plt.plot(theta, Ytrain[i], color=colors[i])
plt.show(block=None)