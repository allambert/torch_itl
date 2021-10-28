import torch
from sklearn.gaussian_process.kernels import RBF
import numpy as np


class SyntheticGPmixture():

    def __init__(self, gamma_cov=((0.05, 0.1, 0.5, 0.7), (0.05, 0.1, 0.5, 0.7)), 
                 noise=(None, None), scale=2):
        self.n_atoms = len(gamma_cov[0])
        self.gamma_cov_input = gamma_cov[0]
        self.gamma_cov_output = gamma_cov[1]
        self.noise_input = noise[0]
        self.noise_output = noise[1]
        self.scale = scale

    def drawGP(self, t, seed_gp=75):
        np.random.seed(seed_gp)
        self.t = t.view(-1, 1)
        # draw GP input
        self.GP_input = torch.zeros(self.n_atoms, t.shape[0])
        for i in range(self.n_atoms):
            kernel = RBF(self.gamma_cov_input[i])
            covmat = kernel(np.expand_dims(t.numpy(), axis=1),
                            np.expand_dims(t.numpy(), axis=1))
            if self.noise_input is not None:
                covmat += np.eye(len(covmat)) * self.noise_input ** 2

            self.GP_input[i] = torch.from_numpy(
                np.random.multivariate_normal(np.zeros(len(covmat)), covmat))
        # draw GP output
        self.GP_output = torch.zeros(self.n_atoms, t.shape[0])
        for i in range(self.n_atoms):
            kernel = RBF(self.gamma_cov_output[i])
            covmat = kernel(np.expand_dims(t.numpy(), axis=1),
                            np.expand_dims(t.numpy(), axis=1))
            if self.noise_output is not None:
                covmat += np.eye(len(covmat)) * self.noise_output ** 2

            self.GP_output[i] = torch.from_numpy(
                np.random.multivariate_normal(np.zeros(len(covmat)), covmat))

    def sample(self, n_samples, new_GP=False, seed_gp=75, seed_coefs=765):
        if not hasattr(self, 'GP_input') or new_GP:
            self.drawGP(self.t, seed_gp)
        np.random.seed(seed_coefs)
        coefficients = torch.from_numpy(self.scale * (np.random.rand(n_samples, self.n_atoms) - 0.5))
        X = coefficients @ self.GP_input
        Y = coefficients @ self.GP_output
        return X, Y


def make_gaussian(t, gamma_cov, noise):
    kernel = RBF(gamma_cov)
    covmat = kernel(np.expand_dims(t.numpy(), axis=1),
                    np.expand_dims(t.numpy(), axis=1))
    if noise is not None:
        covmat += np.eye(len(covmat)) * noise ** 2

    data = np.random.multivariate_normal(np.zeros(len(covmat)), covmat)

    return torch.from_numpy(data)


def synthetic_gaussian(n_samples, t, n_atoms=4, gamma_cov=[0.1, 0.1],
                       noise=[None, None], scale=2):
    noise_input, noise_output = noise
    gp_input = torch.zeros(n_atoms, t.shape[0])
    gp_output = torch.zeros(n_atoms, t.shape[0])
    for i in range(n_atoms):
        gamma_cov_input, gamma_cov_output = gamma_cov[i]
        gp_input[i] = make_gaussian(t, gamma_cov_input, noise_input)
        gp_output[i] = make_gaussian(t, gamma_cov_output, noise_output)

    coefficients = scale * (torch.rand(n_samples, n_atoms) - 0.5)

    X = coefficients @ gp_input
    Y = coefficients @ gp_output

    return X, Y



# n = 20
# t = torch.linspace(0, 1, 64)
# n_atoms_list = [4, 8]
# gamma_cov_list = [np.array([[0.07, 0.07], [0.5, 0.5],
#                             [0.1, 0.1], [0.7, 0.7]]),
#                   np.array([[0.01, 0.01], [0.05, 0.05],
#                             [0.1, 0.1], [0.7, 0.7],
#                             [0.01, 0.01], [0.05, 0.05],
#                             [0.1, 0.1], [0.7, 0.7]])]
# scale_list = [1.5]
# colors = [cm.viridis(x) for x in torch.linspace(0, 1, n)]
# for i, n_atoms in enumerate(n_atoms_list):
#     gamma_cov = gamma_cov_list[i]
#     for scale in scale_list:
#         X, Y = synthetic_gaussian(n, t, n_atoms=n_atoms,
#                                   gamma_cov=gamma_cov, scale=scale)
#         plt.figure()
#         plt.subplot(211)
#         plt.title('Input functions $(x_i)_{i=1}^n$')
#         for i in range(n):
#             plt.plot(t, X[i], color=colors[i])
#         plt.subplot(212)
#         plt.tight_layout()
#         plt.title('Output functions $(y_i)_{i=1}^n$')
#         for i in range(n):
#             plt.plot(t, Y[i], color=colors[i])
#         plt.show(block=None)
