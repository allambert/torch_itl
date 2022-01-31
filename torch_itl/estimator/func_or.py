"""Implement functional output regression classes."""
from abc import ABC, abstractmethod
import torch
from slycot import sb04qd
import time
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from .utils import (proj_vect_2, proj_vect_inf, proj_matrix_2,
                    proj_matrix_inf, bst_matrix, bst_vector, st)
from sklearn.model_selection import KFold


class FuncOR(ABC):
    """Implements Functional Output Regression
    """

    def __init__(self, model, lbda):
        self.model = model
        self.lbda = lbda
        self.losses = None

    @abstractmethod
    def dual_loss_diff(self, alpha):
        pass

    @abstractmethod
    def dual_grad(self, alpha):
        pass

    @abstractmethod
    def prox_step(self, alpha, gamma=None):
        pass

    # To initialize computation of eigen related quantities for eigen solver
    @abstractmethod
    def initialize_specifics(self):
        pass

    # Dimitri: dejà dit dans model mais c'est pas terrible d'initializer des attributs de classe
    # en dehors de la classe, dans le code j'ai déporté cette méthode initialize dans DecomposableIdentityScalar et DecomposableIntOp
    # Pour que ça run avec le reste de ta librairie je ne l'ai pas fait ici mais je conseille
    def initialize(self, x, y, thetas, warm_start=True, requires_grad=True):
        self.model.x_train = x
        self.model.n = len(x)
        self.model.m = len(thetas)
        self.model.y_train = y
        self.model.thetas = thetas
        self.model.initialize(x, warm_start, requires_grad)
        self.model.compute_gram_train()
        self.initialize_specifics()

    def get_rescale_cste(self):
        return self.lbda * self.model.n * self.model.m

    def prox_gd_init(self, x, y, thetas, warm_start, reinit_losses=True):
        self.initialize(x, y, thetas, warm_start=warm_start, requires_grad=False)
        # Losses initialization
        if self.losses is None:
            self.losses = []
        if reinit_losses:
            self.losses = []
        cste = self.get_rescale_cste()
        return self.model.alpha.detach().clone() * cste

    def acc_prox_lsearch(self, t0, alpha_v, grad_v, beta=0.2):
        t = t0
        stop = False
        while not stop:
            alpha_plus = self.prox_step(alpha_v - t * grad_v, t)
            term1 = self.dual_loss_diff(alpha_plus)
            term21 = self.dual_loss_diff(alpha_v)
            term22 = (grad_v * (alpha_plus - alpha_v)).sum()
            term23 = 0.5 * (1 / t) * ((alpha_plus - alpha_v) ** 2).sum()
            term2 = term21 + term22 + term23
            if term1 > term2:
                t *= beta
            else:
                stop = True
        return t

    def fit_acc_prox_restart_gd(self, x, y, thetas, n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
                                monitor_loss=False, reinit_losses=True, d=20):
        alpha = self.prox_gd_init(x, y, thetas, warm_start, reinit_losses=reinit_losses)
        alpha_minus1 = alpha
        alpha_minus2 = alpha
        step_size = 1
        epoch_restart = 0
        converged = False
        for epoch in range(0, n_epoch):
            acc_cste = epoch_restart / (epoch_restart + 1 + d)
            alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
            grad_v = self.dual_grad(alpha_v)
            step_size = self.acc_prox_lsearch(step_size, alpha_v, grad_v, beta)
            alpha_tentative = self.prox_step(alpha_v - step_size * grad_v, step_size)
            if ((alpha_v - alpha_tentative) * (alpha_tentative - alpha_minus1)).sum() > 0:
                print("RESTART")
                epoch_restart = 0
                grad_v = self.dual_grad(alpha_minus1)
                step_size = self.acc_prox_lsearch(step_size, alpha_minus1, grad_v, beta)
                alpha = self.prox_step(alpha_minus1 - step_size * grad_v, step_size)
            else:
                alpha = alpha_tentative
            if monitor_loss:
                self.losses.append(self.dual_loss_diff(alpha))
            diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
            print(diff)
            if diff < tol:
                converged = True
                break
            alpha_minus2 = alpha_minus1.detach().clone()
            alpha_minus1 = alpha.detach().clone()
            epoch_restart += 1
        # Scale alpha to match prediction formula
        cste = self.get_rescale_cste()
        self.model.alpha = alpha / cste
        if not converged:
            raise ConvergenceWarning("Maximum number of iteration reached")

    def predict(self, X, theta):
        return self.model.forward(X, theta)

    def risk(self, X, Y, thetas):
        pred = self.predict(X, thetas)
        return ((Y - pred) ** 2).mean()

    def training_risk(self):
        pred = self.predict(self.model.x_train, self.model.thetas)
        return ((self.model.y_train - pred) ** 2).mean()

    def fit(self, x, y, thetas, n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
            monitor_loss=False, reinit_losses=True, d=20):
        self.fit_acc_prox_restart_gd(x, y, thetas, n_epoch, warm_start, tol,
                                     beta, monitor_loss, reinit_losses)


class FuncORSplines(FuncOR):

    def __init__(self, model, lbda):
        super().__init__(model, lbda)

    def dual_loss_diff(self, alpha):
        A = 0.5 * alpha @ alpha.T
        B = - alpha @ self.model.y_train.T
        cste = 0.5 / (self.lbda * self.model.n * self.model.m)
        C = cste * self.model.G_x @ alpha @ self.model.G_t @ alpha.T
        return torch.trace(A + B + C)

    def dual_grad(self, alpha):
        A = alpha
        B = - self.model.y_train
        cste = 1 / (self.lbda * self.model.n * self.model.m)
        C = cste * self.model.G_x @ alpha @ self.model.G_t
        return A + B + C

    def prox_step(self, alpha, gamma=None):
        return alpha

    def fit_sylvester(self, x, y, thetas):
        self.initialize(x, y, thetas, warm_start=False, requires_grad=False)
        alpha = sb04qd(self.model.n, self.model.m,
                       self.model.G_x.numpy() / (self.lbda * self.model.n * self.model.m),
                       self.model.G_t.numpy(), y.numpy() / (self.lbda * self.model.n * self.model.m))
        self.model.alpha = torch.from_numpy(alpha)

    def fit(self, x, y, thetas):
        self.fit_sylvester(x, y, thetas)


class FuncOREigen(FuncOR):

    def __init__(self, model, lbda):
        super().__init__(model, lbda)

    def dual_loss_diff(self, alpha):
        A = 0.5 * alpha @ alpha.T
        B = - alpha @ self.model.R.T
        cste = 0.5 / (self.lbda * self.model.n * self.model.m)
        C = cste * self.model.G_x @ alpha @ torch.diag(self.model.eig_vals) @ alpha.T
        return torch.trace(A + B + C)

    def dual_grad(self, alpha):
        A = alpha
        B = - self.model.R
        cste = 1 / (self.lbda * self.model.n * self.model.m)
        C = cste * self.model.G_x @ alpha @ torch.diag(self.model.eig_vals)
        return A + B + C

    def prox_step(self, alpha, gamma=None):
        return alpha

    def initialize_specifics(self):
        self.model.compute_eigen_output()
        self.model.compute_R()

    def fit_sylvester(self, x, y, thetas):
        self.initialize(x, y, thetas, warm_start=False, requires_grad=False)
        Lambda = torch.diag(self.model.eig_vals)
        alpha = sb04qd(self.model.n, self.model.n_eigen,
                       self.model.G_x.numpy() / (self.lbda * self.model.n * self.model.m), Lambda.numpy(),
                       self.model.R.numpy() / (self.lbda * self.model.n * self.model.m))
        self.model.alpha = torch.from_numpy(alpha)

    def fit(self, x, y, thetas):
        self.fit_sylvester(x, y, thetas)


class RobustFuncOREigen(FuncOREigen):

    def __init__(self, model, lbda, loss_param=0.1):
        super().__init__(model, lbda)
        self.loss_param = loss_param

    def prox_step(self, alpha, gamma=None):
        return proj_matrix_2(alpha, self.loss_param)

    def fit(self, x, y, thetas, init_sylvester=True,
            n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
            monitor_loss=False, reinit_losses=True, d=20):
        """
        init_sylvester will not work if warm_start is set to False
        """
        if init_sylvester:
            self.fit_sylvester(x, y, thetas)
            if not warm_start:
                raise Warning("Initialization with Sylvester does not work if warm_start==False")
        self.fit_acc_prox_restart_gd(x, y, thetas, n_epoch, warm_start, tol, beta,
                                     monitor_loss, reinit_losses, d)

    # def get_kappa_max(self, alpha, rescale_alpha=False):
    #     n, m = self.model.alpha.shape[0], self.model.alpha.shape[1]
    #     if rescale_alpha:
    #         cste = 1 / (self.lbda * self.model.n * self.model.m)
    #         alpha_sc = alpha * cste
    #     else:
    #         alpha_sc = alpha
    #     pred = self.model.G_x @ alpha_sc @ torch.diag(self.model.eig_vals) @ self.model.eig_vecs
    #     return (1 / np.sqrt(m)) * torch.sqrt(((self.model.y_train - pred) ** 2).sum(dim=1)).max()

    def fit_sylvester(self, x, y, thetas):
        raise ValueError("Sylvester solver only works for square loss, please use FISTA based solvers")

class SparseFuncOREigen(FuncOREigen):

    def __init__(self, model, lbda, loss_param=0.1):
        super().__init__(model, lbda)
        self.loss_param = loss_param

    def prox_step(self, alpha, gamma=None):
        return bst_matrix(alpha, gamma * np.sqrt(self.model.m) * self.loss_param)

    def get_sparsity_level(self):
        n_zeros = len(torch.where(self.model.alpha == 0)[0])
        return n_zeros / (self.model.n_eigen * self.model.n)

    def fit(self, x, y, thetas, init_sylvester=True,
            n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
            monitor_loss=False, reinit_losses=True, d=20):
        """
        init_sylvester will not work if warm_start is set to False
        """
        if init_sylvester:
            self.fit_sylvester(x, y, thetas)
            if not warm_start:
                raise Warning("Initialization with Sylvester does not work if warm_start==False")
        self.fit_acc_prox_restart_gd(x, y, thetas, n_epoch, warm_start, tol, beta,
                                     monitor_loss, reinit_losses, d)


class RobustFuncORSplines(FuncORSplines):

    def __init__(self, model, lbda, loss_param=0.1, norm='inf'):
        super().__init__(model, lbda)
        self.loss_param = loss_param
        self.norm = norm

    def primal_loss(self, alpha, rescale_alpha=True):
        if rescale_alpha:
            cste = 1 / self.get_rescale_cste()
            alpha_sc = alpha * cste
        else:
            alpha_sc = alpha
        pred = self.model.G_x @ alpha_sc @ self.model.G_t
        if self.norm == "2":
            raise ValueError("Primal loss not yet implement for 2 norm")
        elif self.norm == "inf":
            error = self.model.y_train - pred
            error_norms = (1 / np.sqrt(self.model.m)) * torch.sqrt((error ** 2).sum(dim=1))
            mask_sup_kappa = error_norms > self.loss_param
            mask_inf_kappa = error_norms <= self.loss_param
            term_sup_kappa = (self.loss_param * (mask_sup_kappa * (error_norms - 0.5 * self.loss_param))).sum()
            term_inf_kappa = (0.5 * mask_inf_kappa * error_norms ** 2).sum()
            data_fitting = (1 / self.model.n) * (term_inf_kappa + term_sup_kappa)
        else:
            raise ValueError('Not implemented norm')
        regularization = (0.5 * self.lbda / self.model.m ** 2) \
            * torch.trace(self.model.G_x @ alpha_sc @ self.model.G_t @ alpha_sc.T)
        return data_fitting + regularization

    def prox_step(self, alpha, gamma=None):
        if self.norm == '2':
            return proj_matrix_2(alpha, self.loss_param)
        elif self.norm == 'inf':
            return proj_matrix_inf(alpha, self.loss_param)
        else:
            raise ValueError('Not implemented norm')

    def fit(self, x, y, thetas, init_sylvester=True,
            n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
            monitor_loss=False, reinit_losses=True, d=20):
        """
        init_sylvester will not work if warm_start is set to False
        """
        if init_sylvester:
            self.fit_sylvester(x, y, thetas)
            if not warm_start:
                raise Warning("Initialization with Sylvester does not work if warm_start==False")
        self.fit_acc_prox_restart_gd(x, y, thetas, n_epoch, warm_start, tol, beta,
                                     monitor_loss, reinit_losses, d)

    # def get_kappa_max(self, alpha, rescale_alpha=True):
    #     n, m = self.model.alpha.shape[0], self.model.alpha.shape[1]
    #     if rescale_alpha:
    #         cste = 1 / (self.lbda * self.model.n * self.model.m)
    #         alpha_sc = alpha * cste
    #     else:
    #         alpha_sc = alpha
    #     pred = self.model.G_x @ alpha_sc @ self.model.G_t
    #     if self.norm == '2':
    #         return (1 / np.sqrt(m)) * torch.sqrt(((self.model.y_train - pred) ** 2).sum(dim=1)).max()
    #     elif self.norm == 'inf':
    #         return (self.model.y_train - pred).abs().max()
    #     else:
    #         raise ValueError('Not implemented norm')


class SparseFuncORSplines(FuncORSplines):

    def __init__(self, model, lbda, loss_param=0.1, norm='inf'):
        super().__init__(model, lbda)
        self.loss_param = loss_param
        self.norm = norm

    def dual_loss_full(self, alpha):
        if self.norm == "2":
            dual_penalty = np.sqrt(self.model.m) * self.model.alpha.norm(dim=1).sum()
        elif self.norm == "inf":
            dual_penalty = alpha.abs().sum()
        else:
            raise ValueError('Not implemented norm')
        return self.dual_loss_diff(alpha) + self.loss_param * dual_penalty

    def primal_loss(self, alpha, rescale_alpha=True):
        if rescale_alpha:
            cste = 1 / self.get_rescale_cste()
            alpha_sc = alpha * cste
        else:
            alpha_sc = alpha
        error = self.model.y_train - self.model.G_x @ alpha_sc @ self.model.G_t
        if self.norm == "2":
            error_norms = (1 / np.sqrt(self.model.m)) * torch.sqrt((error ** 2).sum(dim=1))
            data_fitting = error_norms.mean()
        elif self.norm == "inf":
            abs_error = error.abs()
            data_fitting = (torch.maximum(abs_error, torch.tensor(self.loss_param)) ** 2).mean()
        else:
            raise ValueError('Not implemented norm')
        regularization = (0.5 * self.lbda / self.model.m ** 2) \
            * torch.trace(self.model.G_x @ alpha_sc @ self.model.G_t @ alpha_sc.T)
        return data_fitting + regularization

    def prox_step(self, alpha, gamma=None):
        if self.norm == '2':
            return bst_matrix(alpha, gamma * np.sqrt(self.model.m) * self.loss_param)
        elif self.norm == 'inf':
            return st(alpha, gamma * self.loss_param)

    def fit(self, x, y, thetas, init_sylvester=True,
            n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
            monitor_loss=False, reinit_losses=True, d=20):
        """
        init_sylvester will not work if warm_start is set to False
        """
        if init_sylvester:
            self.fit_sylvester(x, y, thetas)
            if not warm_start:
                raise Warning("Initialization with Sylvester does not work if warm_start==False")
        self.fit_acc_prox_restart_gd(x, y, thetas, n_epoch, warm_start, tol, beta,
                                     monitor_loss, reinit_losses, d)

    def get_sparsity_level(self):
        n_zeros = len(torch.where(self.model.alpha == 0)[0])
        return n_zeros / (self.model.m * self.model.n)

# class FOR(VITL):
#     """Implements Functional Output Regression
#     """

#     def __init__(self, model, lbda, sampler):
#         super().__init__(model, squared_norm, lbda, sampler)

#     def dual_loss(self):
#         A = 0.5 * self.model.alpha @ self.model.alpha.T
#         B = - self.model.alpha @ self.model.R.T
#         C = 1/(2 * self.lbda * self.model.n) * self.model.G_x \
#             @ self.model.alpha @ self.model.Lambda @ self.model.alpha.T

#         return torch.trace(A + B + C)

#     def dual_grad(self, i=None):
#         if i is None:
#             A = self.model.alpha
#             B = - self.model.R
#             C = 1/(self.lbda * self.model.n) * self.model.G_x \
#                 @ self.model.alpha @ self.model.Lambda
#             return A + B + C
#         else:
#             A = self.model.alpha[i]
#             B = - self.model.R[i]
#             C = 1/(self.lbda * self.model.n) * (self.model.G_x[i]
#                                                 @ self.model.alpha
#                                                 @ self.model.Lambda)
#             return A + B + C

#     def get_stepsize(self, dim=0):
#         if self.norm == '2':
#             normalization_factor = self.lbda*self.model.n
#         else:
#             normalization_factor = self.lbda*self.model.n*self.model.m
#         if dim == 0:
#             lipschitz = 1/ normalization_factor \
#                 * self.model.G_x.trace()
#             if self.norm == '2':
#                 lipschitz *= self.model.Lambda[0, 0]
#             else:
#                 lipschitz *= self.model.G_t.trace()
#             lipschitz += 1
#         elif dim == 1:
#             lipschitz = torch.Tensor([1 / normalization_factor *
#                                       self.model.G_x[i, i]
#                                       for i in range(self.model.n)])
#             if self.norm == '2':
#                 lipschitz *= self.model.Lambda[0, 0]
#             else:
#                 lipschitz *= self.model.G_t.trace()
#                 lipschitz += 1
#         elif dim == 2:
#             lipschitz = 1 / normalization_factor * \
#                 torch.Tensor([[self.model.G_x[i, i] * self.model.G_t[j, j]
#                                for j in range(self.model.m)]
#                               for i in range(self.model.n)])
#             lipschitz += 1
#         else:
#             raise ValueError('Wrong number of dimensions')
#         stepsize = 1 / lipschitz
#         return stepsize

#     def initialize(self, x, y, thetas, warm_start=True, requires_grad=True):
#         self.model.x_train = x
#         self.model.n = x.shape[0]
#         self.model.y_train = y
#         self.model.thetas = thetas
#         self.model.compute_gram_train()
#         self.model.compute_R(thetas, y)
#         self.model.m = self.model.kernel_output.m
#         self.model.initialize(x, warm_start, requires_grad)

#     def fit(self, x, y, thetas, n_epoch=500, solver=torch.optim.LBFGS,
#             warm_start=True, tol=1e-5, **kwargs):
#         self.initialize(x, y, thetas, warm_start=warm_start)
#         if not hasattr(self, 'losses'):
#             self.losses = []
#             self.times = [0]
#         optimizer = torch.optim.LBFGS([self.model.alpha], **kwargs)

#         def closure_alpha():
#             # for i in range(self.model.n):
#             #     optimizer_list[i].zero_grad()
#             optimizer.zero_grad()
#             loss = self.dual_loss()
#             loss.backward()
#             return(loss)

#         t0 = time.time()

#         for epoch in range(n_epoch):
#             loss = closure_alpha()
#             self.losses.append(loss.item())
#             self.times.append(time.time() - t0)
#             optimizer.step(closure_alpha)
#             if ((epoch+5) % 20) == 0:
#                 if (self.losses[-1] - self.losses[-2]).abs() < tol:
#                     break

#         with torch.no_grad():
#             self.model.alpha /= self.lbda * self.model.n

#     def fit_gd(self, x, y, thetas, n_epoch=5000, warm_start=True, tol=1e-5):
#         self.initialize(x, y, thetas,
#                         warm_start=warm_start, requires_grad=False)
#         if warm_start and hasattr(self, 'losses'):
#             self.model.alpha *= self.lbda * self.model.n
#         lipschitz = 1/(self.lbda*self.model.n) * self.model.G_x.trace()
#         lipschitz *= self.model.Lambda[0, 0]
#         lipschitz += 1
#         stepsize = 1/lipschitz

#         if not hasattr(self, 'losses'):
#             self.losses = []
#             self.times = [0]

#         t0 = time.time()
#         for epoch in range(n_epoch):
#             self.model.alpha -= stepsize * self.dual_grad()
#             self.losses.append(self.dual_loss())
#             self.times.append(time.time() - t0)
#             if ((epoch+5) % 20) == 0:
#                 if (self.losses[-1] - self.losses[-2]).abs() < tol:
#                     break

#         self.model.alpha /= self.lbda * self.model.n

#     def fit_cd(self, x, y, thetas, n_epoch=5000, warm_start=True, tol=1e-5):
#         self.initialize(x, y, thetas,
#                         warm_start=warm_start, requires_grad=False)
#         if warm_start and hasattr(self, 'losses'):
#             self.model.alpha *= self.lbda * self.model.n
#         lipschitz = torch.Tensor([1/(self.lbda*self.model.n) *
#                                   self.model.G_x[i, i]
#                                   for i in range(self.model.n)])
#         lipschitz *= self.model.Lambda[0, 0]
#         lipschitz += 1
#         stepsize = 1/lipschitz

#         if not hasattr(self, 'losses'):
#             self.losses = []
#             self.times = [0]

#         t0 = time.time()
#         for epoch in range(n_epoch):
#             for i in range(self.model.n):
#                 self.model.alpha[i] -= stepsize[i] * self.dual_grad(i)
#             self.losses.append(self.dual_loss())
#             self.times.append(time.time() - t0)
#             if ((epoch+5) % 20) == 0:
#                 if (self.losses[-1] - self.losses[-2]).abs() < tol:
#                     break

#         self.model.alpha /= self.lbda * self.model.n

#     def get_kappa_max(self, norm):
#         if norm == '2':
#             return ((self.model.alpha**2).sum(axis=1).sqrt().max()
#                     * self.lbda * self.model.n)
#         elif norm == 'inf':
#             return self.model.alpha.abs().max() \
#                 * self.lbda * self.model.n * self.model.m
#         else:
#             raise ValueError('Not implemented norm')

#     def risk(self, X, Y, thetas):
#         pred = self.predict(X, thetas)
#         return ((Y - pred)**2).mean()

#     def training_risk(self):
#         pred = self.predict(self.model.x_train, self.model.thetas)
#         return ((self.model.y_train - pred)**2).mean()

#     def tune_lambda(self, lbda_grid, X, Y, thetas, n_splits, fit='cd'):
#         mse = torch.zeros(lbda_grid.shape[0], n_splits)
#         for i, lbda in enumerate(lbda_grid):
#             self.lbda = lbda
#             kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#             mse_local = []
#             for train_index, test_index in kf.split(X):
#                 X_train, X_test = X[train_index], X[test_index]
#                 Y_train, Y_test = Y[train_index], Y[test_index]
#                 if fit == 'cd':
#                     self.fit_cd(X_train, Y_train, thetas, n_epoch=3000,
#                                 warm_start=False)
#                 elif fit == 'gd':
#                     self.fit_gd(X_train, Y_train, thetas, n_epoch=3000,
#                                 warm_start=False)
#                 else:
#                     raise ValueError('Unknown fitting procedure provided')
#                 mse_local.append(self.risk(X_test, Y_test, thetas))
#             mse[i] = torch.Tensor(mse_local)
#         lbda_argmin = mse.mean(1).argmin()
#         return lbda_grid[lbda_argmin], mse.mean(1)[lbda_argmin], mse

# class RobustFOR(FOR):
#     """Implements Robust Functional Output Regression
#     """

#     def __init__(self, model, lbda, sampler, kappa=0.1, norm='2'):
#         super().__init__(model, lbda, sampler)
#         self.kappa = kappa
#         self.norm = norm

#     def initialize(self, x, y, thetas, warm_start=True, requires_grad=True):
#         self.model.x_train = x
#         self.model.n = x.shape[0]
#         self.model.y_train = y
#         self.model.thetas = thetas
#         self.model.compute_gram_train()
#         if self.norm == '2':
#             self.model.compute_R(thetas, y)
#             self.model.m = self.model.kernel_output.m
#         if self.norm == 'inf':
#             self.model.m = thetas.shape[0]
#         self.model.initialize(x, warm_start, requires_grad)

#     def dual_loss(self):
#         if self.norm == '2':
#             A = 0.5 * self.model.alpha @ self.model.alpha.T
#             B = - self.model.alpha @ self.model.R.T
#             C = 1/(2 * self.lbda * self.model.n) * self.model.G_x \
#                 @ self.model.alpha @ self.model.Lambda @ self.model.alpha.T
#         elif self.norm == 'inf':
#             A = 0.5 * self.model.alpha @ self.model.alpha.T
#             B = - self.model.alpha @ self.model.y_train.T
#             C = 1/(2 * self.lbda * self.model.n * self.model.m) \
#                 * self.model.G_x @ self.model.alpha @ self.model.G_t \
#                 @ self.model.alpha.T
#         else:
#             raise ValueError('Not implemented norm')
#         return torch.trace(A + B + C)

#     def dual_grad(self, i=None, j=None):
#         if i is None:
#             if self.norm == '2':
#                 A = self.model.alpha
#                 B = - self.model.R
#                 C = 1/(self.lbda * self.model.n) * self.model.G_x \
#                     @ self.model.alpha @ self.model.Lambda
#             elif self.norm == 'inf':
#                 A = self.model.alpha
#                 B = - self.model.y_train
#                 C = 1/(self.lbda * self.model.n * self.model.m) \
#                     * self.model.G_x @ self.model.alpha @ self.model.G_t
#             else:
#                 raise ValueError('Not implemented norm')
#             return A + B + C
#         elif j is None:
#             if self.norm == '2':
#                 A = self.model.alpha[i]
#                 B = - self.model.R[i]
#                 C = 1/(self.lbda * self.model.n) * self.model.G_x[i] \
#                     @ self.model.alpha @ self.model.Lambda
#             elif self.norm == 'inf':
#                 A = self.model.alpha[i]
#                 B = - self.model.y_train[i]
#                 C = 1/(self.lbda * self.model.n * self.model.m) \
#                     * self.model.G_x[i] \
#                     @ self.model.alpha @ self.model.G_t
#             else:
#                 raise ValueError('Not implemented norm')
#             return A + B + C
#         else:
#             if self.norm == '2':
#                 A = self.model.alpha[i, j]
#                 B = - self.model.R[i, j]
#                 C = 1/(self.lbda * self.model.n) * self.model.G_x[i] \
#                     @ self.model.alpha @ self.model.Lambda[j]
#             elif self.norm == 'inf':
#                 A = self.model.alpha[i, j]
#                 B = - self.model.y_train[i, j]
#                 C = 1/(self.lbda * self.model.n * self.model.m) \
#                     * self.model.G_x[i] \
#                     @ self.model.alpha @ self.model.G_t[j]
#             else:
#                 raise ValueError('Not implemented norm')
#             return A + B + C

#     def proximal_step(self, i=None, j=None):
#         if self.norm == '2':
#             self.model.alpha = proj_matrix_2(self.model.alpha, self.kappa)
#         elif self.norm == 'inf':
#             if i is not None and j is not None:
#                 self.model.alpha[i, j] = torch.where(
#                     self.model.alpha[i, j].abs() > self.kappa,
#                     torch.Tensor([self.kappa]),
#                     self.model.alpha[i, j])
#             else:
#                 self.model.alpha = proj_matrix_inf(self.model.alpha,
#                                                    self.kappa)
#         else:
#             raise ValueError('Not implemented norm')

#     def fit(self, x, y, thetas, n_epoch=50, solver=torch.optim.LBFGS,
#             warm_start=True, tol=1e-5, **kwargs):
#         self.initialize(x, y, thetas)
#         if not hasattr(self, 'losses'):
#             self.losses = []
#             self.times = [0]

#         optimizer = torch.optim.LBFGS([self.model.alpha], **kwargs)

#         def closure_alpha():
#             optimizer.zero_grad()
#             loss = self.dual_loss()
#             loss.backward()
#             return(loss)

#         t0 = time.time()

#         for epoch in range(n_epoch):
#             loss = closure_alpha()
#             self.losses.append(loss.item())
#             self.times.append(time.time() - t0)
#             optimizer.step(closure_alpha)
#             with torch.no_grad():
#                 if self.norm == '2':
#                     for i in range(self.model.n):
#                         self.model.alpha[i] = proj_vect_2(self.model.alpha[i],
#                                                           self.kappa)
#                 elif self.norm == 'inf':
#                     for i in range(self.model.n):
#                         self.model.alpha[i] = proj_vect_inf(self.model.alpha[i],
#                                                             self.kappa)
#                 else:
#                     raise ValueError('Not implemented norm')
#             if ((epoch+5) % 20) == 0:
#                 if (self.losses[-1] - self.losses[-2]).abs() < tol:
#                     break

#         with torch.no_grad():
#             self.model.alpha /= self.lbda * self.model.n

#     def fit_gd(self, x, y, thetas, n_epoch=2000, warm_start=True, tol=1e-5):
#         self.initialize(x, y, thetas,
#                         warm_start=warm_start, requires_grad=False)
#         if self.norm == '2':
#             factor = 1.
#         else:
#             factor = self.model.m
#         if warm_start and hasattr(self.model, 'alpha'):
#             self.model.alpha *= self.lbda * self.model.n * factor
#         lipschitz = 1/(self.lbda*self.model.n * factor) \
#             * self.model.G_x.trace()
#         if self.norm == '2':
#             lipschitz *= self.model.Lambda[0, 0]
#         elif self.norm == 'inf':
#             lipschitz *= self.model.G_t.trace()
#         lipschitz += 1
#         stepsize = 1/lipschitz

#         if not hasattr(self, 'losses'):
#             self.losses = []
#             self.times = [0]

#         t0 = time.time()
#         for epoch in range(n_epoch):
#             self.model.alpha -= stepsize * self.dual_grad()
#             self.proximal_step()
#             self.losses.append(self.dual_loss())
#             self.times.append(time.time() - t0)
#             if ((epoch+5) % 20) == 0:
#                 if (self.losses[-1] - self.losses[-2]).abs() \
#                         / self.losses[-1] < tol:
#                     break

#         if self.norm == '2':
#             self.model.alpha /= self.lbda * self.model.n
#         elif self.norm == 'inf':
#             self.model.alpha /= self.lbda * self.model.n * self.model.m

#     def fit_cd(self, x, y, thetas, n_epoch=50, warm_start=True, tol=1e-5):
#         self.initialize(x, y, thetas,
#                         warm_start=warm_start, requires_grad=False)
#         if warm_start and hasattr(self, 'losses'):
#             self.model.alpha *= self.lbda * self.model.n
#         lipschitz = torch.Tensor([1/(self.lbda*self.model.n) *
#                                   self.model.G_x[i, i]
#                                   for i in range(self.model.n)])
#         if self.norm == '2':
#             lipschitz *= self.model.Lambda[0, 0]
#         elif self.norm == 'inf':
#             lipschitz *= self.model.G_t.trace()
#         lipschitz += 1
#         stepsize = 1/lipschitz

#         if not hasattr(self, 'losses'):
#             self.losses = []
#             self.times = [0]

#         t0 = time.time()
#         for epoch in range(n_epoch):
#             for i in range(self.model.n):
#                 if self.norm == '2':
#                     self.model.alpha[i] -= stepsize[i] * self.dual_grad(i)
#                     self.proximal_step()
#                 elif self.norm == 'inf':
#                     for j in range(self.model.m):
#                         self.model.alpha[i, j] -= stepsize[i] \
#                             * self.dual_grad(i, j)
#                         self.proximal_step(i, j)
#             self.losses.append(self.dual_loss())
#             self.times.append(time.time() - t0)
#             if ((epoch+5) % 20) == 0:
#                 if (self.losses[-1] - self.losses[-2]).abs() < tol:
#                     break
#         if self.norm == '2':
#             self.model.alpha /= self.lbda * self.model.n
#         elif self.norm == 'inf':
#             self.model.alpha /= self.lbda * self.model.n * self.model.m

#     def saturated_constraints(self, locs=False):
#         if self.norm == '2':
#             tmp = torch.isclose(self.lbda * self.model.n *
#                                 torch.sqrt(torch.sum(self.model.alpha**2,
#                                                      axis=1)),
#                                 torch.Tensor([self.kappa]))
#             if locs:
#                 return (tmp.sum().float()/self.model.n, torch.where(tmp == 1))
#             else:
#                 return tmp.sum().float()/self.model.n

#         if self.norm == 'inf':
#             tmp = torch.isclose(self.lbda * self.model.n * self.model.m *
#                                 torch.abs(self.model.alpha),
#                                 torch.Tensor([self.kappa]))
#             if locs:
#                 return (tmp.sum().float()/self.model.n/self.model.m,
#                         torch.where(tmp == 1))
#             else:
#                 return tmp.sum().float()/self.model.n/self.model.m


# class SparseFOR(FOR):
#     """Implements sparse Functional Output Regression
#     """

#     def __init__(self, model, lbda, sampler, epsilon=0.1, norm='2'):
#         super().__init__(model, lbda, sampler)
#         self.epsilon = epsilon
#         self.norm = norm

#     def initialize(self, x, y, thetas, warm_start=True, requires_grad=True):
#         self.model.x_train = x
#         self.model.n = x.shape[0]
#         self.model.y_train = y
#         self.model.thetas = thetas
#         self.model.compute_gram_train()
#         if self.norm == '2':
#             self.model.compute_R(thetas, y)
#             self.model.m = self.model.kernel_output.m
#         if self.norm == 'inf':
#             self.model.m = thetas.shape[0]
#         self.model.initialize(x, warm_start, requires_grad)

#     def dual_loss(self):
#         if self.norm == '2':
#             A = 0.5 * self.model.alpha @ self.model.alpha.T
#             B = - self.model.alpha @ self.model.R.T
#             C = 1/(2 * self.lbda * self.model.n) * self.model.G_x \
#                 @ self.model.alpha @ self.model.Lambda @ self.model.alpha.T
#             pen = self.epsilon * (self.model.alpha ** 2).sum(1).sqrt().sum()
#         elif self.norm == 'inf':
#             A = 0.5 * self.model.alpha @ self.model.alpha.T
#             B = - self.model.alpha @ self.model.y_train.T
#             C = 1/(2 * self.lbda * self.model.n * self.model.m) \
#                 * self.model.G_x @ self.model.alpha @ self.model.G_t \
#                 @ self.model.alpha.T
#             pen = self.epsilon * self.model.alpha.abs().sum()
#         else:
#             raise ValueError('Not implemented norm')
#         return pen + torch.trace(A + B + C)

#     def dual_grad(self, i=None, j=None):
#         if i is None:
#             if self.norm == '2':
#                 A = self.model.alpha
#                 B = - self.model.R
#                 C = 1/(self.lbda * self.model.n) * self.model.G_x \
#                     @ self.model.alpha @ self.model.Lambda
#             elif self.norm == 'inf':
#                 A = self.model.alpha
#                 B = - self.model.y_train
#                 C = 1/(self.lbda * self.model.n * self.model.m) \
#                     * self.model.G_x @ self.model.alpha @ self.model.G_t
#             else:
#                 raise ValueError('Not implemented norm')
#             return A + B + C
#         elif j is None:
#             if self.norm == '2':
#                 A = self.model.alpha[i]
#                 B = - self.model.R[i]
#                 C = 1/(self.lbda * self.model.n) * self.model.G_x[i] \
#                     @ self.model.alpha @ self.model.Lambda
#             elif self.norm == 'inf':
#                 A = self.model.alpha[i]
#                 B = - self.model.y_train[i]
#                 C = 1/(self.lbda * self.model.n * self.model.m) \
#                     * self.model.G_x[i] \
#                     @ self.model.alpha @ self.model.G_t
#             else:
#                 raise ValueError('Not implemented norm')
#             return A + B + C
#         else:
#             if self.norm == '2':
#                 A = self.model.alpha[i, j]
#                 B = - self.model.R[i, j]
#                 C = 1/(self.lbda * self.model.n) * self.model.G_x[i] \
#                     @ self.model.alpha @ self.model.Lambda[j]
#             elif self.norm == 'inf':
#                 A = self.model.alpha[i, j]
#                 B = - self.model.y_train[i, j]
#                 C = 1/(self.lbda * self.model.n * self.model.m) \
#                     * self.model.G_x[i] \
#                     @ self.model.alpha @ self.model.G_t[j]
#             else:
#                 raise ValueError('Not implemented norm')
#             return A + B + C

#     def proximal_step(self, stepsize, i=None, j=None):
#         if i is None:
#             if self.norm == '2':
#                 self.model.alpha = bst_matrix(self.model.alpha, stepsize *
#                                               self.epsilon)
#             elif self.norm == 'inf':
#                 self.model.alpha = st(self.model.alpha,
#                                       stepsize * self.epsilon)
#         elif j is None:
#             if self.norm == '2':
#                 self.model.alpha[i] = bst_vector(self.model.alpha[i],
#                                                  stepsize * self.epsilon)
#             elif self.norm == 'inf':
#                 self.model.alpha[i] = st(self.model.alpha[i],
#                                          stepsize * self.epsilon)
#         else:
#             if self.norm == 'inf':
#                 self.model.alpha[i, j] = st(self.model.alpha[i, j],
#                                             stepsize * self.epsilon)

#     def fit_gd(self, x, y, thetas, n_epoch=2000, warm_start=True, tol=1e-5):
#         self.initialize(x, y, thetas,
#                         warm_start=warm_start, requires_grad=False)
#         if self.norm == '2':
#             factor = 1.
#         else:
#             factor = self.model.m
#         if warm_start and hasattr(self.model, 'alpha'):
#             self.model.alpha *= self.lbda * self.model.n * factor
#         lipschitz = 1/(self.lbda*self.model.n * factor) \
#             * self.model.G_x.trace()
#         if self.norm == '2':
#             lipschitz *= self.model.Lambda[0, 0]
#         elif self.norm == 'inf':
#             lipschitz *= self.model.G_t.trace()
#         lipschitz += 1
#         stepsize = 1/lipschitz

#         if not hasattr(self, 'losses'):
#             self.losses = []
#             self.times = [0]

#         t0 = time.time()
#         for epoch in range(n_epoch):
#             self.model.alpha -= stepsize * self.dual_grad()
#             self.proximal_step(stepsize)
#             self.losses.append(self.dual_loss())
#             self.times.append(time.time() - t0)
#             # if ((epoch+5) % 20) == 0:
#             #     if (self.losses[-1] - self.losses[-2]).abs() \
#             #             / self.losses[-1].abs() < tol:
#             #         break

#         if self.norm == '2':
#             self.model.alpha /= self.lbda * self.model.n
#         elif self.norm == 'inf':
#             self.model.alpha /= self.lbda * self.model.n * self.model.m

#     def fit_cd(self, x, y, thetas, n_epoch=50, warm_start=True, tol=1e-5):
#         self.initialize(x, y, thetas,
#                         warm_start=warm_start, requires_grad=False)
#         if warm_start and hasattr(self, 'losses'):
#             if self.norm == '2':
#                 self.model.alpha *= self.lbda * self.model.n
#             elif self.norm == 'inf':
#                 self.model.alpha *= self.lbda * self.model.n * self.model.m

#         if self.norm == 'inf':
#             stepsize = self.get_stepsize(dim=2)
#         else:
#             stepsize = self.get_stepsize(dim=1)

#         if not hasattr(self, 'losses'):
#             self.losses = []
#             self.times = [0]

#         t0 = time.time()
#         for epoch in range(n_epoch):
#             for i in range(self.model.n):
#                 if self.norm == '2':
#                     self.model.alpha[i] -= stepsize[i] * self.dual_grad(i)
#                     self.proximal_step(stepsize[i], i)
#                 elif self.norm == 'inf':
#                     for j in range(self.model.m):
#                         self.model.alpha[i, j] -= stepsize[i, j] \
#                             * self.dual_grad(i, j)
#                         self.proximal_step(stepsize[i, j], i, j)
#             self.losses.append(self.dual_loss())
#             self.times.append(time.time() - t0)
#             if ((epoch+5) % 20) == 0:
#                 if (self.losses[-1] - self.losses[-2]).abs() < tol:
#                     break

#         if self.norm == '2':
#             self.model.alpha /= self.lbda * self.model.n
#         elif self.norm == 'inf':
#             self.model.alpha /= self.lbda * self.model.n * self.model.m

#     def saturated_constraints(self, locs=False):
#         if self.norm == '2':
#             tmp = torch.isclose(self.lbda * self.model.n *
#                                 torch.sqrt(torch.sum(self.model.alpha**2,
#                                                      axis=1)),
#                                 torch.Tensor([0]))
#             if locs:
#                 return (tmp.sum().float()/self.model.n, torch.where(tmp == 1))
#             else:
#                 return tmp.sum().float()/self.model.n

#         if self.norm == 'inf':
#             tmp = torch.isclose(self.lbda * self.model.n * self.model.m *
#                                 torch.abs(self.model.alpha),
#                                 torch.Tensor([0]))
#             if locs:
#                 return (tmp.sum().float()/self.model.n/self.model.m,
#                         torch.where(tmp == 1))
#             else:
#                 return tmp.sum().float()/self.model.n/self.model.m
