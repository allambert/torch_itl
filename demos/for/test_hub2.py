import torch
import numpy as np
import matplotlib. pyplot as plt
import matplotlib.cm as cm

from torch_itl.datasets import SyntheticGPmixture
from torch_itl.model import DecomposableIntOp
from torch_itl.kernel import Gaussian
from torch_itl.estimator import FuncOREigen, RobustFuncOREigen

# Torch default datatype
torch.set_default_dtype(torch.float64)


# Generate synthetic dataset ###########################################
# ######################################################################
n_train = 100
n_test = 100
seed_train = 765
seed_test = 443
# Sampling locations
thetas = torch.linspace(0, 1, 100)
# Generate data
data_synth = SyntheticGPmixture()
data_synth.drawGP(thetas)
Xtrain, Ytrain = data_synth.sample(n_train, seed_coefs=seed_train)
Xtest, Ytest = data_synth.sample(n_test, seed_coefs=seed_test)

# Plot dataset
n_plot = n_train
colors = [cm.viridis(x) for x in np.linspace(0, 1, n_plot)]
plt.figure()
plt.subplot(211)
plt.title('Input functions $(x_i)_{i=1}^n$')
for i in range(n_plot):
    plt.plot(thetas, Xtrain[i], color=colors[i])
plt.subplot(212)
plt.tight_layout()
plt.title('Output functions $(y_i)_{i=1}^n$')
for i in range(n_plot):
    plt.plot(thetas, Ytrain[i], color=colors[i])
plt.show(block=None)


# Define models, estimators and test fit and predict ###################
# ######################################################################
# kernel_input = LaplacianSum(5)
kernel_input = Gaussian(0.01)
# kernel_output = Gaussian(40)
kernel_output = Gaussian(100)
lbda = 1e-7


# Square loss
n_eigen = 25
mod = DecomposableIntOp(kernel_input, kernel_output, n_eigen=n_eigen)
esti = FuncOREigen(mod, lbda)
esti.fit(Xtrain, Ytrain, thetas.view(-1, 1))
pred = esti.predict(Xtest, thetas.view(-1, 1))
# Print test score
print(((pred - Ytest) ** 2).mean())


# Epsilon insensitive 2 loss
kappa = 0.05
esti_hub2 = RobustFuncOREigen(mod, lbda, loss_param=kappa)
esti_hub2.fit(Xtrain, Ytrain, thetas.view(-1, 1))
pred_hub2 = esti_hub2.predict(Xtest, thetas.view(-1, 1))
# Print test score
print(((pred_hub2 - Ytest) ** 2).mean())