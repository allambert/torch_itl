import torch
import time
from .utils import (squared_norm, proj_vect_2, proj_vect_inf, proj_matrix_2,
                    proj_matrix_inf, bst_matrix, bst_vector, st)
from .vitl import VITL
from sklearn.model_selection import KFold


class FOR(VITL):
    """Implements Functional Output Regression
    """

    def __init__(self, model, lbda, sampler):
        super().__init__(model, squared_norm, lbda, sampler)

    def dual_loss(self):
        A = 0.5 * self.model.alpha @ self.model.alpha.T
        B = - self.model.alpha @ self.model.R.T
        C = 1/(2 * self.lbda * self.model.n) * self.model.G_x \
            @ self.model.alpha @ self.model.Lambda @ self.model.alpha.T

        return torch.trace(A + B + C)

    def dual_grad(self, i=None):
        if i is None:
            A = self.model.alpha
            B = - self.model.R
            C = 1/(self.lbda * self.model.n) * self.model.G_x \
                @ self.model.alpha @ self.model.Lambda
            return A + B + C
        else:
            A = self.model.alpha[i]
            B = - self.model.R[i]
            C = 1/(self.lbda * self.model.n) * (self.model.G_x[i]
                                                @ self.model.alpha
                                                @ self.model.Lambda)
            return A + B + C

    def get_stepsize(self, dim=0):
        if self.norm == '2':
            normalization_factor = self.lbda*self.model.n
        else:
            normalization_factor = self.lbda*self.model.n*self.model.m
        if dim == 0:
            lipschitz = 1/ normalization_factor \
                * self.model.G_x.trace()
            if self.norm == '2':
                lipschitz *= self.model.Lambda[0, 0]
            else:
                lipschitz *= self.model.G_t.trace()
            lipschitz += 1
        elif dim == 1:
            lipschitz = torch.Tensor([1 / normalization_factor *
                                      self.model.G_x[i, i]
                                      for i in range(self.model.n)])
            if self.norm == '2':
                lipschitz *= self.model.Lambda[0, 0]
            else:
                lipschitz *= self.model.G_t.trace()
                lipschitz += 1
        elif dim == 2:
            lipschitz = 1 / normalization_factor * \
                torch.Tensor([[self.model.G_x[i, i] * self.model.G_t[j, j]
                               for j in range(self.model.m)]
                              for i in range(self.model.n)])
            lipschitz += 1
        else:
            raise ValueError('Wrong number of dimensions')
        stepsize = 1 / lipschitz
        return stepsize

    def initialise(self, x, y, thetas, warm_start=True, requires_grad=True):
        self.model.x_train = x
        self.model.n = x.shape[0]
        self.model.y_train = y
        self.model.thetas = thetas
        self.model.compute_gram_train()
        self.model.compute_R(thetas, y)
        self.model.m = self.model.kernel_output.m
        self.model.initialise(x, warm_start, requires_grad)

    def fit(self, x, y, thetas, n_epoch=500, solver=torch.optim.LBFGS,
            warm_start=True, tol=1e-5, **kwargs):
        self.initialise(x, y, thetas, warm_start=warm_start)
        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]
        optimizer = torch.optim.LBFGS([self.model.alpha], **kwargs)

        def closure_alpha():
            # for i in range(self.model.n):
            #     optimizer_list[i].zero_grad()
            optimizer.zero_grad()
            loss = self.dual_loss()
            loss.backward()
            return(loss)

        t0 = time.time()

        for epoch in range(n_epoch):
            loss = closure_alpha()
            self.losses.append(loss.item())
            self.times.append(time.time() - t0)
            optimizer.step(closure_alpha)
            if ((epoch+5) % 20) == 0:
                if (self.losses[-1] - self.losses[-2]).abs() < tol:
                    break

        with torch.no_grad():
            self.model.alpha /= self.lbda * self.model.n

    def fit_gd(self, x, y, thetas, n_epoch=5000, warm_start=True, tol=1e-5):
        self.initialise(x, y, thetas,
                        warm_start=warm_start, requires_grad=False)
        if warm_start and hasattr(self, 'losses'):
            self.model.alpha *= self.lbda * self.model.n
        lipschitz = 1/(self.lbda*self.model.n) * self.model.G_x.trace()
        lipschitz *= self.model.Lambda[0, 0]
        lipschitz += 1
        stepsize = 1/lipschitz

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        t0 = time.time()
        for epoch in range(n_epoch):
            self.model.alpha -= stepsize * self.dual_grad()
            self.losses.append(self.dual_loss())
            self.times.append(time.time() - t0)
            if ((epoch+5) % 20) == 0:
                if (self.losses[-1] - self.losses[-2]).abs() < tol:
                    break

        self.model.alpha /= self.lbda * self.model.n

    def fit_cd(self, x, y, thetas, n_epoch=5000, warm_start=True, tol=1e-5):
        self.initialise(x, y, thetas,
                        warm_start=warm_start, requires_grad=False)
        if warm_start and hasattr(self, 'losses'):
            self.model.alpha *= self.lbda * self.model.n
        lipschitz = torch.Tensor([1/(self.lbda*self.model.n) *
                                  self.model.G_x[i, i]
                                  for i in range(self.model.n)])
        lipschitz *= self.model.Lambda[0, 0]
        lipschitz += 1
        stepsize = 1/lipschitz

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        t0 = time.time()
        for epoch in range(n_epoch):
            for i in range(self.model.n):
                self.model.alpha[i] -= stepsize[i] * self.dual_grad(i)
            self.losses.append(self.dual_loss())
            self.times.append(time.time() - t0)
            if ((epoch+5) % 20) == 0:
                if (self.losses[-1] - self.losses[-2]).abs() < tol:
                    break

        self.model.alpha /= self.lbda * self.model.n

    def get_kappa_max(self, norm):
        if norm == '2':
            return ((self.model.alpha**2).sum(axis=1).sqrt().max()
                    * self.lbda * self.model.n)
        elif norm == 'inf':
            return self.model.alpha.abs().max() \
                * self.lbda * self.model.n * self.model.m
        else:
            raise ValueError('Not implemented norm')

    def risk(self, X, Y, thetas):
        pred = self.predict(X, thetas)
        return ((Y - pred)**2).mean()

    def training_risk(self):
        pred = self.predict(self.model.x_train, self.model.thetas)
        return ((self.model.y_train - pred)**2).mean()

    def tune_lambda(self, lbda_grid, X, Y, thetas, n_splits, fit='cd'):
        mse = torch.zeros(lbda_grid.shape[0], n_splits)
        for i, lbda in enumerate(lbda_grid):
            self.lbda = lbda
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            mse_local = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                if fit == 'cd':
                    self.fit_cd(X_train, Y_train, thetas, n_epoch=3000,
                                warm_start=False)
                elif fit == 'gd':
                    self.fit_gd(X_train, Y_train, thetas, n_epoch=3000,
                                warm_start=False)
                else:
                    raise ValueError('Unknown fitting procedure provided')
                mse_local.append(self.risk(X_test, Y_test, thetas))
            mse[i] = torch.Tensor(mse_local)
        lbda_argmin = mse.mean(1).argmin()
        return lbda_grid[lbda_argmin], mse.mean(1)[lbda_argmin], mse


class RobustFOR(FOR):
    """Implements Robust Functional Output Regression
    """

    def __init__(self, model, lbda, sampler, kappa=0.1, norm='2'):
        super().__init__(model, lbda, sampler)
        self.kappa = kappa
        self.norm = norm

    def initialise(self, x, y, thetas, warm_start=True, requires_grad=True):
        self.model.x_train = x
        self.model.n = x.shape[0]
        self.model.y_train = y
        self.model.thetas = thetas
        self.model.compute_gram_train()
        if self.norm == '2':
            self.model.compute_R(thetas, y)
            self.model.m = self.model.kernel_output.m
        if self.norm == 'inf':
            self.model.m = thetas.shape[0]
        self.model.initialise(x, warm_start, requires_grad)

    def dual_loss(self):
        if self.norm == '2':
            A = 0.5 * self.model.alpha @ self.model.alpha.T
            B = - self.model.alpha @ self.model.R.T
            C = 1/(2 * self.lbda * self.model.n) * self.model.G_x \
                @ self.model.alpha @ self.model.Lambda @ self.model.alpha.T
        elif self.norm == 'inf':
            A = 0.5 * self.model.alpha @ self.model.alpha.T
            B = - self.model.alpha @ self.model.y_train.T
            C = 1/(2 * self.lbda * self.model.n * self.model.m) \
                * self.model.G_x @ self.model.alpha @ self.model.G_t \
                @ self.model.alpha.T
        else:
            raise ValueError('Not implemented norm')
        return torch.trace(A + B + C)

    def dual_grad(self, i=None, j=None):
        if i is None:
            if self.norm == '2':
                A = self.model.alpha
                B = - self.model.R
                C = 1/(self.lbda * self.model.n) * self.model.G_x \
                    @ self.model.alpha @ self.model.Lambda
            elif self.norm == 'inf':
                A = self.model.alpha
                B = - self.model.y_train
                C = 1/(self.lbda * self.model.n * self.model.m) \
                    * self.model.G_x @ self.model.alpha @ self.model.G_t
            else:
                raise ValueError('Not implemented norm')
            return A + B + C
        elif j is None:
            if self.norm == '2':
                A = self.model.alpha[i]
                B = - self.model.R[i]
                C = 1/(self.lbda * self.model.n) * self.model.G_x[i] \
                    @ self.model.alpha @ self.model.Lambda
            elif self.norm == 'inf':
                A = self.model.alpha[i]
                B = - self.model.y_train[i]
                C = 1/(self.lbda * self.model.n * self.model.m) \
                    * self.model.G_x[i] \
                    @ self.model.alpha @ self.model.G_t
            else:
                raise ValueError('Not implemented norm')
            return A + B + C
        else:
            if self.norm == '2':
                A = self.model.alpha[i, j]
                B = - self.model.R[i, j]
                C = 1/(self.lbda * self.model.n) * self.model.G_x[i] \
                    @ self.model.alpha @ self.model.Lambda[j]
            elif self.norm == 'inf':
                A = self.model.alpha[i, j]
                B = - self.model.y_train[i, j]
                C = 1/(self.lbda * self.model.n * self.model.m) \
                    * self.model.G_x[i] \
                    @ self.model.alpha @ self.model.G_t[j]
            else:
                raise ValueError('Not implemented norm')
            return A + B + C

    def proximal_step(self, i=None, j=None):
        if self.norm == '2':
            self.model.alpha = proj_matrix_2(self.model.alpha, self.kappa)
        elif self.norm == 'inf':
            if i is not None and j is not None:
                self.model.alpha[i, j] = torch.where(
                    self.model.alpha[i, j].abs() > self.kappa,
                    torch.Tensor([self.kappa]),
                    self.model.alpha[i, j])
            else:
                self.model.alpha = proj_matrix_inf(self.model.alpha,
                                                   self.kappa)
        else:
            raise ValueError('Not implemented norm')

    def fit(self, x, y, thetas, n_epoch=50, solver=torch.optim.LBFGS,
            warm_start=True, tol=1e-5, **kwargs):
        self.initialise(x, y, thetas)
        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        optimizer = torch.optim.LBFGS([self.model.alpha], **kwargs)

        def closure_alpha():
            optimizer.zero_grad()
            loss = self.dual_loss()
            loss.backward()
            return(loss)

        t0 = time.time()

        for epoch in range(n_epoch):
            loss = closure_alpha()
            self.losses.append(loss.item())
            self.times.append(time.time() - t0)
            optimizer.step(closure_alpha)
            with torch.no_grad():
                if self.norm == '2':
                    for i in range(self.model.n):
                        self.model.alpha[i] = proj_vect_2(self.model.alpha[i],
                                                          self.kappa)
                elif self.norm == 'inf':
                    for i in range(self.model.n):
                        self.model.alpha[i] = proj_vect_inf(self.model.alpha[i],
                                                            self.kappa)
                else:
                    raise ValueError('Not implemented norm')
            if ((epoch+5) % 20) == 0:
                if (self.losses[-1] - self.losses[-2]).abs() < tol:
                    break

        with torch.no_grad():
            self.model.alpha /= self.lbda * self.model.n

    def fit_gd(self, x, y, thetas, n_epoch=2000, warm_start=True, tol=1e-5):
        self.initialise(x, y, thetas,
                        warm_start=warm_start, requires_grad=False)
        if self.norm == '2':
            factor = 1.
        else:
            factor = self.model.m
        if warm_start and hasattr(self.model, 'alpha'):
            self.model.alpha *= self.lbda * self.model.n * factor
        lipschitz = 1/(self.lbda*self.model.n * factor) \
            * self.model.G_x.trace()
        if self.norm == '2':
            lipschitz *= self.model.Lambda[0, 0]
        elif self.norm == 'inf':
            lipschitz *= self.model.G_t.trace()
        lipschitz += 1
        stepsize = 1/lipschitz

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        t0 = time.time()
        for epoch in range(n_epoch):
            self.model.alpha -= stepsize * self.dual_grad()
            self.proximal_step()
            self.losses.append(self.dual_loss())
            self.times.append(time.time() - t0)
            if ((epoch+5) % 20) == 0:
                if (self.losses[-1] - self.losses[-2]).abs() \
                        / self.losses[-1] < tol:
                    break

        if self.norm == '2':
            self.model.alpha /= self.lbda * self.model.n
        elif self.norm == 'inf':
            self.model.alpha /= self.lbda * self.model.n * self.model.m

    def fit_cd(self, x, y, thetas, n_epoch=50, warm_start=True, tol=1e-5):
        self.initialise(x, y, thetas,
                        warm_start=warm_start, requires_grad=False)
        if warm_start and hasattr(self, 'losses'):
            self.model.alpha *= self.lbda * self.model.n
        lipschitz = torch.Tensor([1/(self.lbda*self.model.n) *
                                  self.model.G_x[i, i]
                                  for i in range(self.model.n)])
        if self.norm == '2':
            lipschitz *= self.model.Lambda[0, 0]
        elif self.norm == 'inf':
            lipschitz *= self.model.G_t.trace()
        lipschitz += 1
        stepsize = 1/lipschitz

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        t0 = time.time()
        for epoch in range(n_epoch):
            for i in range(self.model.n):
                if self.norm == '2':
                    self.model.alpha[i] -= stepsize[i] * self.dual_grad(i)
                    self.proximal_step()
                elif self.norm == 'inf':
                    for j in range(self.model.m):
                        self.model.alpha[i, j] -= stepsize[i] \
                            * self.dual_grad(i, j)
                        self.proximal_step(i, j)
            self.losses.append(self.dual_loss())
            self.times.append(time.time() - t0)
            if ((epoch+5) % 20) == 0:
                if (self.losses[-1] - self.losses[-2]).abs() < tol:
                    break
        if self.norm == '2':
            self.model.alpha /= self.lbda * self.model.n
        elif self.norm == 'inf':
            self.model.alpha /= self.lbda * self.model.n * self.model.m

    def saturated_constraints(self, locs=False):
        if self.norm == '2':
            tmp = torch.isclose(self.lbda * self.model.n *
                                torch.sqrt(torch.sum(self.model.alpha**2,
                                                     axis=1)),
                                torch.Tensor([self.kappa]))
            if locs:
                return (tmp.sum().float()/self.model.n, torch.where(tmp == 1))
            else:
                return tmp.sum().float()/self.model.n

        if self.norm == 'inf':
            tmp = torch.isclose(self.lbda * self.model.n * self.model.m *
                                torch.abs(self.model.alpha),
                                torch.Tensor([self.kappa]))
            if locs:
                return (tmp.sum().float()/self.model.n/self.model.m,
                        torch.where(tmp == 1))
            else:
                return tmp.sum().float()/self.model.n/self.model.m


class SparseFOR(FOR):
    """Implements sparse Functional Output Regression
    """

    def __init__(self, model, lbda, sampler, epsilon=0.1, norm='2'):
        super().__init__(model, lbda, sampler)
        self.epsilon = epsilon
        self.norm = norm

    def initialise(self, x, y, thetas, warm_start=True, requires_grad=True):
        self.model.x_train = x
        self.model.n = x.shape[0]
        self.model.y_train = y
        self.model.thetas = thetas
        self.model.compute_gram_train()
        if self.norm == '2':
            self.model.compute_R(thetas, y)
            self.model.m = self.model.kernel_output.m
        if self.norm == 'inf':
            self.model.m = thetas.shape[0]
        self.model.initialise(x, warm_start, requires_grad)

    def dual_loss(self):
        if self.norm == '2':
            A = 0.5 * self.model.alpha @ self.model.alpha.T
            B = - self.model.alpha @ self.model.R.T
            C = 1/(2 * self.lbda * self.model.n) * self.model.G_x \
                @ self.model.alpha @ self.model.Lambda @ self.model.alpha.T
            pen = self.epsilon * (self.model.alpha ** 2).sum(1).sqrt().sum()
        elif self.norm == 'inf':
            A = 0.5 * self.model.alpha @ self.model.alpha.T
            B = - self.model.alpha @ self.model.y_train.T
            C = 1/(2 * self.lbda * self.model.n * self.model.m) \
                * self.model.G_x @ self.model.alpha @ self.model.G_t \
                @ self.model.alpha.T
            pen = self.epsilon * self.model.alpha.abs().sum()
        else:
            raise ValueError('Not implemented norm')
        return pen + torch.trace(A + B + C)

    def dual_grad(self, i=None, j=None):
        if i is None:
            if self.norm == '2':
                A = self.model.alpha
                B = - self.model.R
                C = 1/(self.lbda * self.model.n) * self.model.G_x \
                    @ self.model.alpha @ self.model.Lambda
            elif self.norm == 'inf':
                A = self.model.alpha
                B = - self.model.y_train
                C = 1/(self.lbda * self.model.n * self.model.m) \
                    * self.model.G_x @ self.model.alpha @ self.model.G_t
            else:
                raise ValueError('Not implemented norm')
            return A + B + C
        elif j is None:
            if self.norm == '2':
                A = self.model.alpha[i]
                B = - self.model.R[i]
                C = 1/(self.lbda * self.model.n) * self.model.G_x[i] \
                    @ self.model.alpha @ self.model.Lambda
            elif self.norm == 'inf':
                A = self.model.alpha[i]
                B = - self.model.y_train[i]
                C = 1/(self.lbda * self.model.n * self.model.m) \
                    * self.model.G_x[i] \
                    @ self.model.alpha @ self.model.G_t
            else:
                raise ValueError('Not implemented norm')
            return A + B + C
        else:
            if self.norm == '2':
                A = self.model.alpha[i, j]
                B = - self.model.R[i, j]
                C = 1/(self.lbda * self.model.n) * self.model.G_x[i] \
                    @ self.model.alpha @ self.model.Lambda[j]
            elif self.norm == 'inf':
                A = self.model.alpha[i, j]
                B = - self.model.y_train[i, j]
                C = 1/(self.lbda * self.model.n * self.model.m) \
                    * self.model.G_x[i] \
                    @ self.model.alpha @ self.model.G_t[j]
            else:
                raise ValueError('Not implemented norm')
            return A + B + C

    def proximal_step(self, stepsize, i=None, j=None):
        if i is None:
            if self.norm == '2':
                self.model.alpha = bst_matrix(self.model.alpha, stepsize *
                                              self.epsilon)
            elif self.norm == 'inf':
                self.model.alpha = st(self.model.alpha,
                                      stepsize * self.epsilon)
        elif j is None:
            if self.norm == '2':
                self.model.alpha[i] = bst_vector(self.model.alpha[i],
                                                 stepsize * self.epsilon)
            elif self.norm == 'inf':
                self.model.alpha[i] = st(self.model.alpha[i],
                                         stepsize * self.epsilon)
        else:
            if self.norm == 'inf':
                self.model.alpha[i, j] = st(self.model.alpha[i, j],
                                            stepsize * self.epsilon)

    def fit_gd(self, x, y, thetas, n_epoch=2000, warm_start=True, tol=1e-5):
        self.initialise(x, y, thetas,
                        warm_start=warm_start, requires_grad=False)
        if self.norm == '2':
            factor = 1.
        else:
            factor = self.model.m
        if warm_start and hasattr(self.model, 'alpha'):
            self.model.alpha *= self.lbda * self.model.n * factor
        lipschitz = 1/(self.lbda*self.model.n * factor) \
            * self.model.G_x.trace()
        if self.norm == '2':
            lipschitz *= self.model.Lambda[0, 0]
        elif self.norm == 'inf':
            lipschitz *= self.model.G_t.trace()
        lipschitz += 1
        stepsize = 1/lipschitz

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        t0 = time.time()
        for epoch in range(n_epoch):
            self.model.alpha -= stepsize * self.dual_grad()
            self.proximal_step(stepsize)
            self.losses.append(self.dual_loss())
            self.times.append(time.time() - t0)
            # if ((epoch+5) % 20) == 0:
            #     if (self.losses[-1] - self.losses[-2]).abs() \
            #             / self.losses[-1].abs() < tol:
            #         break

        if self.norm == '2':
            self.model.alpha /= self.lbda * self.model.n
        elif self.norm == 'inf':
            self.model.alpha /= self.lbda * self.model.n * self.model.m

    def fit_cd(self, x, y, thetas, n_epoch=50, warm_start=True, tol=1e-5):
        self.initialise(x, y, thetas,
                        warm_start=warm_start, requires_grad=False)
        if warm_start and hasattr(self, 'losses'):
            if self.norm == '2':
                self.model.alpha *= self.lbda * self.model.n
            elif self.norm == 'inf':
                self.model.alpha *= self.lbda * self.model.n * self.model.m

        if self.norm == 'inf':
            stepsize = self.get_stepsize(dim=2)
        else:
            stepsize = self.get_stepsize(dim=1)

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        t0 = time.time()
        for epoch in range(n_epoch):
            for i in range(self.model.n):
                if self.norm == '2':
                    self.model.alpha[i] -= stepsize[i] * self.dual_grad(i)
                    self.proximal_step(stepsize[i], i)
                elif self.norm == 'inf':
                    for j in range(self.model.m):
                        self.model.alpha[i, j] -= stepsize[i, j] \
                            * self.dual_grad(i, j)
                        self.proximal_step(stepsize[i, j], i, j)
            self.losses.append(self.dual_loss())
            self.times.append(time.time() - t0)
            if ((epoch+5) % 20) == 0:
                if (self.losses[-1] - self.losses[-2]).abs() < tol:
                    break

        if self.norm == '2':
            self.model.alpha /= self.lbda * self.model.n
        elif self.norm == 'inf':
            self.model.alpha /= self.lbda * self.model.n * self.model.m

    def saturated_constraints(self, locs=False):
        if self.norm == '2':
            tmp = torch.isclose(self.lbda * self.model.n *
                                torch.sqrt(torch.sum(self.model.alpha**2,
                                                     axis=1)),
                                torch.Tensor([0]))
            if locs:
                return (tmp.sum().float()/self.model.n, torch.where(tmp == 1))
            else:
                return tmp.sum().float()/self.model.n

        if self.norm == 'inf':
            tmp = torch.isclose(self.lbda * self.model.n * self.model.m *
                                torch.abs(self.model.alpha),
                                torch.Tensor([0]))
            if locs:
                return (tmp.sum().float()/self.model.n/self.model.m,
                        torch.where(tmp == 1))
            else:
                return tmp.sum().float()/self.model.n/self.model.m