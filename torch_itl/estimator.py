import torch
import torch.optim as optim
import time

dtype = torch.float

def kron(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
        [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))

def inverse_reg_block(K1, K2, K3, lbda):
    u1, s1, v1 = torch.svd(K1)
    u2, s2, v2 = torch.svd(K2)
    u3, s3, v3 = torch.svd(K3)
    u = kron(u1.contiguous(), kron(u2.contiguous(), u3.contiguous()))
    s = kron(s1.view(-1,1), kron(s2.view(-1,1), s3.view(-1,1))).view(-1)
    inv_s = 1/ (s + lbda)
    return u @ torch.diag(inv_s) @ u.T

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

    def training_risk(self):
        pred = self.model.forward(self.model.x_train, self.model.thetas)
        return self.cost(self.model.y_train, pred, self.model.thetas)

    def fit_closed_form(self, x_train, y_train):
        n = x_train.shape[0]
        if not hasattr(self.sampler, 'm'):
            self.model.m = int(torch.floor(
                torch.sqrt(torch.Tensor([n]))).item())
            self.sampler.m = self.model.m
        else:
            self.model.m = self.sampler.m
        self.model.thetas = self.sampler.sample(self.model.m)
        self.model.n = n

        self.model.x_train = x_train
        self.model.y_train = y_train

        if not hasattr(self.model, 'G_x'):
            self.model.precompute_gram()

        tmp = self.model.G_xt + self.lbda *n*self.model.m * torch.eye(n*self.model.m)
        tmp = torch.inverse(tmp)

        #tmp = inverse_reg_block(self.model.G_x, self.model.G_t, self.model.kernel_freq, self.lbda *n*self.model.m)

        self.model.alpha = (tmp @ y_train.reshape(-1,self.model.num_freq)).reshape(n, self.model.m, self.model.num_freq)
        #self.model.alpha = (tmp @ y_train.reshape(-1)).reshape(n, self.model.m, self.model.num_freq)


        print('alpha fitted, empirical risk=', self.training_risk())

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

class ITLEstimatorJoint(object):
    "ITL Class with fitting procedure using pytorch"

    def __init__(self, model, cost, lbda_reg, lbda_id, sampler, mask='all'):
        self.model = model
        self.cost = cost
        self.lbda_reg = lbda_reg
        self.lbda_id = lbda_id
        self.sampler = sampler
        self.model.mask = mask

    def objective(self, x, y, thetas):
        "Computes the objectif function to be minimized, sum of the cost +regularization"
        pred = self.model.forward(x, thetas)
        obj = self.cost(y, pred, thetas).mean()
#        obj += 0.5 * self.lbda_id *
        obj += 0.5 * self.lbda_reg * self.model.vv_norm()
        if self.model.kernel_input.is_learnable:
            obj += self.model.kernel_input.regularization()
        if self.model.kernel_output.is_learnable:
            obj += self.model.kernel_output.regularization()
        return(obj)

    def training_risk(self):
        pred = self.model.forward(self.model.x_train, self.model.thetas)
        return self.cost(self.model.y_train, pred, self.model.thetas)

    def fast_objective(self):
        if not hasattr(self.model, 'G_x'):
            self.model.precompute_gram()

        pred = self.model.fast_forward()
        obj = self.cost(self.model.y_train, pred, self.model.thetas).mean()
        obj += 0.5 * self.lbda_reg * self.model.fast_vv_norm()

        return(obj)

    def fit_alpha(self, data, n_epochs=500, solver='lbfgs', minibatch=False, warm_start=True, **kwargs):
        """Fits the parameters alpha of the model. The matrix of coefficients alpha is obtained using
        LBFGS. If kernel_input or kernel_output are learnable, an optimization pass
        is made on the coefficients of their feature maps, with parameters defined as
        attributes in the kernels
        """
        n, m, nf = data.shape
        x_train = data.reshape(-1, nf)
        y_train = torch.zeros(m*n, m, nf)
        for i in range(m*n):
            y_train[i] = data[i//m]

        self.model.n = n*m
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.initialise(x_train, y_train, warm_start)

        if self.model.mask == 'all':
            self.model.mask = torch.ones(n*m)

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

    def fast_fit_alpha(self, data, n_epochs=500, solver='lbfgs', minibatch=False, warm_start=True, **kwargs):
        """Fits the parameters alpha of the model. The matrix of coefficients alpha is obtained using
        LBFGS. If kernel_input or kernel_output are learnable, an optimization pass
        is made on the coefficients of their feature maps, with parameters defined as
        attributes in the kernels
        """
        m = data.shape[1]
        x_train = data.reshape(-1, self.model.num_freq)
        y_train = data.expand(m, -1, -1, -1).reshape(-1, m, self.model.num_freq)
        n = x_train.shape[0]

        self.model.n = n
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.initialise(x_train, y_train, warm_start)

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
            loss = self.fast_objective()
            optimizer_alpha.zero_grad()
            loss.backward()
            return(loss)

        t0 = time.time()

        for k in range(n_epochs):
            loss = closure_alpha()
            self.losses.append(loss.item())
            self.times.append(time.time() - t0)
            optimizer_alpha.step(closure_alpha)

    def fit_closed_form(self, data):
        """Fits using a closed form solution - involves inverting several matrices"""

        n, m, nf = data.shape
        x_train = data.reshape(-1, nf)
        y_train = torch.zeros(m*n, m, nf)
        for i in range(m*n):
            y_train[i] = data[i//m]

        self.model.n = n*m
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.x_train = x_train
        self.model.y_train = y_train
        if not hasattr(self.model, 'G_x'):
            self.model.precompute_gram()

        tmp = self.model.G_xt + self.lbda_reg *n*m*m * torch.eye(n*m*m)
        tmp = torch.inverse(tmp)

        # tmp = inverse_reg_block(self.model.G_x, self.model.G_t, self.model.kernel_freq, self.lbda_reg *n*self.model.m)
        # self.model.alpha = (tmp @ y_train.reshape(-1)).reshape(n*m, m, nf)


        self.model.alpha = (tmp @ y_train.reshape(-1,nf)).reshape(n*m, m, nf)

        print('alpha fitted, empirical risk=', self.training_risk())

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
