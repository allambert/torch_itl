import torch
import torch.optim as optim
import time
from .cost import *
from .kernel import *
from .sampler import *
from .model import *

dtype = torch.float


class ITLEstimator(object):
    "ITL Class with fitting procedure using pytorch"

    def __init__(self, model, cost, lbda, sampler):
        self.model = model
        self.cost = cost
        self.lbda = lbda
        self.sampler = sampler

    def objective(self, x, y, thetas):
        "Computes the objectif function to be minimized, sum of the cost +regularization"
        pred = self.model.forward(x, thetas)
        obj = self.cost(y, pred, thetas)
        obj += 0.5 * self.lbda * self.model.vv_norm()
        if self.model.kernel_input.is_learnable:
            obj += self.model.kernel_input.regularization()
        if self.model.kernel_output.is_learnable:
            obj += self.model.kernel_output.regularization()
        return(obj)

    def fit_alpha(self, x_train, y_train, n_epochs=500, solver='lbfgs', minibatch=False, warm_start=True, **kwargs):
        """Fits the parameters alpha of the model. The matrix of coefficients alpha is obtained using
        LBFGS. If kernel_input or kernel_output are learnable, an optimization pass
        is made on the coefficients of their feature maps, with parameters defined as
        attributes in the kernels
        """
        n = x_train.shape[0]
        if not hasattr(self.sampler, 'm'):
            self.model.m = int(torch.floor(
                torch.sqrt(torch.Tensor([n]))).item())
            self.sampler.m = self.model.m
        else:
            self.model.m = self.sampler.m
        self.model.thetas = self.sampler.sample(self.model.m)
        self.model.n = n

        self.model.initialise(x_train, warm_start)

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        if solver == 'lbfgs':
            optimizer_alpha = optim.LBFGS([self.model.alpha], **kwargs)
        elif solver == 'sgd':
            optimizer_alpha = optim.SGD([self.model.alpha], **kwargs)
        elif solver == 'adam':
            optimizer_alpha = optim.Adam([self.model.alpha], **kwargs)

        def closure_alpha():
            loss = self.objective(x_train, y_train, self.model.thetas)
            optimizer_alpha.zero_grad()
            loss.backward()
            return(loss)

        t0 = time.time()

        for k in range(n_epochs):
            loss = closure_alpha()
            self.losses.append(loss.item())
            self.times.append(time.time() - t0)
            optimizer_alpha.step(closure_alpha)

    def fit_kernel_input(self, x_train, y_train, n_epochs=150, solver='sgd', minibatch=False):

        optimizer_kernel = optim.SGD(
            self.model.kernel_input.model.parameters(),
            lr=self.model.kernel_input.optim_params['lr'],
            momentum=self.model.kernel_input.optim_params['momentum'],
            dampening=self.model.kernel_input.optim_params['dampening'],
            weight_decay=self.model.kernel_input.optim_params['weight_decay'],
            nesterov=self.model.kernel_input.optim_params['nesterov'])

        def closure_kernel():
            loss = self.objective(x_train, y_train, self.model.thetas)
            optimizer_kernel.zero_grad()
            loss.backward()
            return loss

        t0 = time.time()

        if not hasattr(self.model.kernel_input, 'losses'):
            self.model.kernel_input.losses = []
            self.model.kernel_input.times = [0]

        for k in range(n_epochs):
            loss = closure_kernel()
            self.model.kernel_input.losses.append(loss.item())
            self.model.kernel_input.times.append(time.time() - t0)
            optimizer_kernel.step(closure_kernel)

    def fit_kernel_output(self, x_train, y_train, n_epochs=150, solver='sgd', minibatch=False):

        optimizer_kernel = optim.SGD(
            self.model.kernel_output.model.parameters,
            lr=self.model.kernel_output.optim_params['lr'],
            momentum=self.model.kernel_output.optim_params['momentum'],
            dampening=self.model.kernel_output.optim_params['dampening'],
            weight_decay=self.model.kernel_output.optim_params['weight_decay'],
            nesterov=self.model.kernel_output.optim_params['nesterov'])

        def closure_kernel():
            loss = self.objective(x_train, y_train, self.model.thetas)
            optimizer_kernel.zero_grad()
            loss.backward()
            return loss

        t0 = time.time()

        if not hasattr(self.model.kernel_output, 'losses'):
            self.model.kernel_output.losses = []
            self.model.kernel_output.times = [0]

        for k in range(n_epochs):
            loss = closure_kernel()
            self.model.kernel_output.losses.append(loss.item())
            self.model.kernel_output.times.append(time.time() - t0)
            optimizer_kernel.step(closure_kernel)

    def clear_memory(self):
        "Clears memory of the model"
        self.losses, self.times = [], [0]
        if self.model.kernel_input.is_learnable:
            self.model.kernel_input.clear_memory()
        if self.model.kernel_output.is_learnable:
            self.model.kernel_output.clear_memory()
