import torch
import torch.optim as optim
import time
import cost
import kernel
import sampler

dtype = torch.float

class ITL(object):
    "ITL Class with fitting procedure using pytorch"

    def __init__(self, kernel_input, kernel_output, cost, lbda, sampler):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.cost = cost
        self.lbda = lbda
        self.sampler = sampler

    def feed(self, x_train, thetas):
        "Set some x_train and thetas to the model, without optimizing"
        self.n = x_train.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x_train
        self.thetas = thetas
        self.alpha = torch.randn((self.n, self.m), requires_grad=True)

    def forward(self, x, thetas):
        "Computes the output of the model at point (x,thetas)"
        if not hasattr(self, 'x_train'):
            raise ValueError('Model not feeded with data')
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(
            self.thetas, thetas)
        return 1 / self.lbda / self.n * G_x @ self.alpha @ G_t

    def vv_norm(self):
        "Computes the vv-RKHS norm of the model"
        G_x = self.kernel_input.compute_gram(self.x_train)
        G_t = self.kernel_output.compute_gram(self.thetas)
        return 1 / self.lbda**2 / self.n**2 * torch.trace(G_x @ self.alpha @ G_t @ self.alpha.T)

    def objective(self, x, y, thetas):
        "Computes the objectif function to be minimized, sum of the cost +regularization"
        pred = self.forward(x, thetas)
        obj = self.cost(y, pred, thetas)
        obj += 0.5 * self.lbda * self.vv_norm()
        if self.kernel_input.is_learnable:
            obj += self.kernel_input.regularization()
        if self.kernel_output.is_learnable:
            obj += self.kernel_output.regularization()
        return(obj)

    def fit_alpha(self, x_train, y_train, n_epochs=500, solver='lbfgs', minibatch=False,**kwargs):
        """Fits the parameters alpha of the model. The matrix of coefficients alpha is obtained using
        LBFGS. If kernel_input or kernel_output are learnable, an optimization pass
        is made on the coefficients of their feature maps, with parameters defined as
        attributes in the kernels
        """
        if not hasattr(self.sampler, 'm'):
            self.m = int(torch.floor(torch.sqrt(torch.Tensor(n))).item())
            self.sampler.m = self.m
        else:
            self.m = self.sampler.m
        self.thetas = self.sampler.sample(self.m)
        self.n = x_train.shape[0]
        self.x_train = x_train
        if not hasattr(self,'alpha'):
            self.alpha = torch.randn((self.n, self.m), requires_grad=True)

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        optimizer_alpha = optim.LBFGS([self.alpha], **kwargs)

        def closure_alpha():
            loss = self.objective(x_train, y_train, self.thetas)
            optimizer_alpha.zero_grad()
            loss.backward()
            return(loss)

        t0 = time.time()

        for k in range(n_epochs):
            loss = closure_alpha()
            self.losses.append(loss.item())
            self.times.append(time.time() - t0)
            optimizer_alpha.step(closure_alpha)

    def fit_kernel_input(self,x_train, y_train,n_epochs=150, solver='sgd', minibatch=False):

        optimizer_kernel = optim.SGD(
            self.kernel_input.model.parameters(),
            lr = self.kernel_input.optim_params['lr'],
            momentum = self.kernel_input.optim_params['momentum'],
            dampening = self.kernel_input.optim_params['dampening'],
            weight_decay = self.kernel_input.optim_params['weight_decay'],
            nesterov = self.kernel_input.optim_params['nesterov'])

        def closure_kernel():
            loss = self.objective(x_train, y_train, self.thetas)
            optimizer_kernel.zero_grad()
            loss.backward()
            return loss

        t0 = time.time()

        if not hasattr(self.kernel_input, 'losses'):
            self.kernel_input.losses = []
            self.kernel_input.times = [0]

        for k in range(n_epochs):
            loss = closure_kernel()
            self.kernel_input.losses.append(loss.item())
            self.kernel_input.times.append(time.time()-t0)
            optimizer_kernel.step(closure_kernel)

    def fit_kernel_output(self,x_train, y_train,n_epochs=150, solver='sgd', minibatch=False):

        optimizer_kernel = optim.SGD(
            self.kernel_output.model.parameters,
            lr = self.kernel_output.optim_params['lr'],
            momentum = self.kernel_output.optim_params['momentum'],
            dampening = self.kernel_output.optim_params['dampening'],
            weight_decay = self.kernel_output.optim_params['weight_decay'],
            nesterov = self.kernel_output.optim_params['nesterov'])

        def closure_kernel():
            loss = self.objective(x_train, y_train, self.thetas)
            optimizer_kernel.zero_grad()
            loss.backward()
            return loss

        t0 = time.time()

        if not hasattr(self.kernel_output, 'losses'):
            self.kernel_output.losses = []
            self.kernel_output.times = [0]

        for k in range(n_epochs):
            loss = closure_kernel()
            self.kernel_output.losses.append(loss.item())
            self.kernel_output.times.append(time.time()-t0)
            optimizer_kernel.step(closure_kernel)

    def clear_memory(self):
        "Clears memory of the model"
        self.losses, self.times = [], [0]
        if self.kernel_input.is_learnable:
            self.kernel_input.clear_memory()
        if self.kernel_output.is_learnable:
            self.kernel_output.clear_memory()
