import torch
from .utils import rbf_kernel, get_anchors_gaussian_rff
from math import pi


class Gaussian(object):

    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        return rbf_kernel(X, Y, self.gamma)


class GaussianSum(object):

    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        if Y is None:
            Y = X
        n_i = X.shape[0]
        n_o = Y.shape[0]
        G = torch.zeros(n_i, n_o)
        for i in range(n_i):
            for j in range(n_o):
                G[i, j] = torch.exp(-self.gamma * (X[i] - Y[j])**2).mean()
        return G


class GaussianRFF(object):

    def __init__(self, dim_input, dim_rff, gamma):
        self.dim_rff = dim_rff
        self.is_learnable = False
        self.anchors = get_anchors_gaussian_rff(
            dim_input, dim_rff, gamma)

    def feature_map(self, X):
        a = torch.cos(X @ self.anchors)
        b = torch.sin(X @ self.anchors)
        return 1 / torch.sqrt(torch.Tensor([self.dim_rff])) * \
            torch.cat((a, b), 1)

    def d_feature_map(self, X):
        a = torch.cos(X @ self.anchors)
        b = torch.sin(X @ self.anchors)
        if self.anchors.shape[0] != 1:
            raise ValueError('theta is not one dimensional')
        c = torch.cat((-b, a), 1)
        coeffs = torch.cat((self.anchors, self.anchors), 1)
        return 1 / torch.sqrt(torch.Tensor([self.dim_rff])) * \
            coeffs * c

    def get_eigen(self):
        phi = self.feature_map(torch.linspace(0, 1, 5000).view(-1, 1))
        covmat = 1./5000 * phi.T @ phi
        u, eigen, _ = torch.svd(covmat)
        self.eigen = eigen * self.dim_rff
        self.u = u
        self.m = eigen.shape[0]
        phi = self.feature_map(torch.linspace(0, 1, 5000).view(-1, 1))
        psi = phi @ u
        self.normalization = torch.sqrt((psi ** 2).mean(0))
        return self.eigen

    def compute_psi(self, X):
        if not hasattr(self, 'eigen'):
            phi = self.feature_map(torch.linspace(0, 1, 2000).view(-1, 1))
            covmat = 1./2000 * phi.T @ phi
            u, eigen = torch.svd(covmat)
            self.eigen = eigen
            self.u = u
        phi = self.feature_map(X)
        return phi @ self.u / self.normalization

    def compute_gram(self, X, Y):
        if Y is None:
            Y = X
        phi_X = self.feature_map(X)
        phi_Y = self.feature_map(Y)
        return phi_X @ phi_Y

class Laplacian(object):

    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        if Y is None:
            Y = X
        n_i = X.shape[0]
        n_o = Y.shape[0]
        G = torch.zeros(n_i, n_o)
        for i in range(n_i):
            for j in range(n_o):
                G[i, j] = torch.exp(-self.gamma * torch.abs(X[i] - Y[j]).sum())
        return G


class LaplacianSum(object):

    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        if Y is None:
            Y = X
        n_i = X.shape[0]
        n_o = Y.shape[0]
        G = torch.zeros(n_i, n_o)
        for i in range(n_i):
            for j in range(n_o):
                G[i, j] = torch.exp(-self.gamma * torch.abs(X[i] - Y[j])).mean()
        return G

class Linear(object):

    def __init__(self):
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        if Y is None:
            Y = X
        return X @ Y.T


class Harmonic(object):

    def __init__(self, m):
        self.m = m
        self.is_learnable = False

    def compute_psi(self, thetas):
        d = thetas.shape[0]
        psi = torch.zeros(self.m, d)
        c = torch.sqrt(torch.Tensor([2]))
        for i in range(self.m//2):
            for j in range(d):
                psi[2*i, j] = c * torch.cos(pi * 2 * i * thetas[j])
                psi[2 * i + 1, j] = c * torch.sin(pi * 2 * i * thetas[j])
        return(psi.T)

    def compute_gram(self, X, Y=None):
        if Y is None:
            Y = X
        psi_x = self.compute_psi(X)
        psi_y = self.compute_psi(Y)
        eigen_matrix = torch.diag(self.get_eigen())
        return psi_x @ eigen_matrix @ psi_y.T

    def get_eigen(self):
        s = torch.zeros(self.m)
        for i in range(self.m//2):
            s[2 * i] = 1 / (1 + i)**2
            s[2 * i + 1] = 1 / (1 + i)**2
        return s

# from math import pi
#
# def f(c):
#     return torch.cos(c)/torch.sin(c) - 0.5*(c - 1/c)
#
#
# def df(c):
#     return -1/torch.sin(c)**2 -0.5 -1/c**2
#
#
# def newton(f, df, c, n_epochs):
#     for i in range(n_epochs):
#         c -= f(c)/df(c)
#     return c
#
#
# def dichotomy(f, c, n_epochs):
#     if n_epochs == 0:
#         return c
#     else:
#         tmp = f(0.5*(c[0] + c[1]))
#         if tmp > 0:
#             c[1] = 0.5*(c[0] + c[1])
#         else:
#             c[0] = 0.5*(c[0] + c[1])
#         print(c)
#         return dichotomy(f, c, n_epochs - 1)
#
#
# # %%
# c = torch.Tensor([k*pi + 0.001 for k in range(50)])
# locs = newton(f, df, c, 100000)
# f(locs)
# locs
# eigen = 2/(1 + locs**2)
# eigen
#
# def compute_psi(thetas):
#     d = thetas.shape[0]
#     psi = torch.zeros(100, d)
#     c = torch.sqrt(torch.Tensor([2]))
#     for i in range(self.m//2):
#         for j in range(d):
#             psi[2*i, j] = c * torch.cos(pi * 2 * i * thetas[j])
#             psi[2 * i + 1, j] = c * torch.sin(pi * 2 * i * thetas[j])
#     return(psi.T)
#
#
# def psi(mu):
#     def _psi(x):
#         return mu*torch.cos(mu*x) + torch.sin(mu*x)
#     return _psi
#
# (psi(locs[0])(torch.linspace(0, 1, 100000))**2).mean()
#
# (psi(locs[3])(torch.linspace(0, 1, 100000))*psi(locs[0])(torch.linspace(0, 1, 100000))).mean()
#
# locs[0]**2 + locs[0]/4*torch.sin(2*locs[0]) - 1/4*torch.cos(2*locs[0]) +0.5 - 1/4/locs[0]*torch.sin(2*locs[0])
# # %%
# c = torch.linspace(0, 50, 10000)
# plt.figure()
# plt.plot(c, f(c))
# plt.ylim(-10, 10)
# plt.show()
