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
    s = kron(s1.view(-1, 1), kron(s2.view(-1, 1), s3.view(-1, 1))).view(-1)
    inv_s = 1 / (s + lbda)
    return u @ torch.diag(inv_s) @ u.T


def proj(alpha, kappa):
    norm = torch.sqrt(torch.sum(alpha**2, axis=1))
    mask = torch.where(norm > kappa)
    alpha[mask] *= kappa / norm[mask].reshape((-1, 1))
    return alpha


def bst(alpha, tau):
    norm = (alpha**2).sum(1).sqrt()
    mask_st = torch.where(norm >= tau)
    mask_ze = torch.where(norm < tau)
    alpha[mask_st] = alpha[mask_st] - alpha[mask_st] / \
        norm[mask_st].reshape((-1, 1)) * tau
    alpha[mask_ze] = 0
    return(alpha)

def iht(alpha, tau):
    return torch.where(alpha.abs() - tau < 0, torch.zeros_like(alpha), alpha.abs() - tau)*torch.sign(alpha)


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

    def risk(self, x, y):
        pred = self.model.forward(x, self.model.thetas)
        return self.cost(y, pred, self.model.thetas)

    def fit_closed_form(self, x_train, y_train, warm_start=False):
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

        if not hasattr(self.model, 'G_x') or not warm_start:
            self.model.precompute_gram()

        tmp = self.model.G_xt + self.lbda * n * \
            self.model.m * torch.eye(n * self.model.m)
        tmp = torch.inverse(tmp)

        #tmp = inverse_reg_block(self.model.G_x, self.model.G_t, self.model.kernel_freq, self.lbda *n*self.model.m)

        self.model.alpha = (tmp @ y_train.reshape(-1, self.model.num_freq)
                            ).reshape(n, self.model.m, self.model.num_freq)
        #self.model.alpha = (tmp @ y_train.reshape(-1)).reshape(n, self.model.m, self.model.num_freq)

        #print('alpha fitted, empirical risk=', self.training_risk())

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

    def __init__(self, model, cost, lbda_reg, lbda_id, sampler):
        self.model = model
        self.cost = cost
        self.lbda_reg = lbda_reg
        self.lbda_id = lbda_id
        self.sampler = sampler

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

    def risk(self, data):
        n, m, nf = data.shape
        x = data.reshape(n * m, nf)
        y = torch.zeros(n * m, m, nf)
        for i in range(n * m):
            y[i] = data[i // m]

        pred = self.model.forward(x, self.model.thetas)
        return self.cost(y, pred, self.model.thetas)

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
        y_train = torch.zeros(m * n, m, nf)
        for i in range(m * n):
            y_train[i] = data[i // m]

        self.model.n = n * m
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.initialise(x_train, y_train, warm_start)

        # if self.model.mask == 'all':
        #     self.model.mask = torch.ones(n*m)

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
        y_train = data.expand(m, -1, -1, -1).reshape(-1,
                                                     m, self.model.num_freq)
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

    def fit_closed_form(self, data, warm_start=False):
        """Fits using a closed form solution - involves inverting several matrices"""

        n, m, nf = data.shape
        x_train = data.reshape(-1, nf)
        y_train = torch.zeros(m * n, m, nf)
        for i in range(m * n):
            y_train[i] = data[i // m]

        # n = self.model.mask.sum().item()
        # _, m, nf = data.shape
        # x_train = data.reshape(-1, nf)[self.model.mask]
        # y_train = torch.zeros(n, m, nf)

        self.model.n = n * m
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.x_train = x_train
        self.model.y_train = y_train
        if not hasattr(self.model, 'G_x') or not warm_start:
            self.model.precompute_gram()

        tmp = self.model.G_xt + self.lbda_reg * \
            n * m * m * torch.eye(n * m * m)
        #tmp = torch.inverse(tmp)
        alpha, _ = torch.solve(y_train.reshape(-1, nf), tmp)
        # tmp = inverse_reg_block(self.model.G_x, self.model.G_t, self.model.kernel_freq, self.lbda_reg *n*self.model.m)
        # self.model.alpha = (tmp @ y_train.reshape(-1)).reshape(n*m, m, nf)

        self.model.alpha = alpha.reshape(n * m, m, nf)
        #self.model.alpha = (tmp @ y_train.reshape(-1,nf)).reshape(n*m, m, nf)

        #print('alpha fitted, empirical risk=', self.training_risk())

    def grad_huber(self, alpha):
        return(self.lbda_reg * alpha + 1 / self.model.n / self.model.m * (self.model.G_xt @ alpha
                                                                          - self.model.y_train.reshape(self.model.n * self.model.m, -1)))

    def huber_loss_reg(self, alpha):
        return 0.5 * torch.trace(self.lbda_reg * alpha @ alpha.T +
                                 1 / self.model.n / self.model.m * (self.model.G_xt @ alpha @ alpha.T -
                                                                    2 * alpha @ self.model.y_train.reshape(self.model.n * self.model.m, -1).T))

    def fit_huber(self, data, kappa, stepsize, n_iter, warm_start=False):

        n, m, nf = data.shape
        x_train = data.reshape(-1, nf)
        y_train = torch.zeros(m * n, m, nf)
        for i in range(m * n):
            y_train[i] = data[i // m]

        self.model.n = n * m
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.x_train = x_train
        self.model.y_train = y_train
        if not hasattr(self.model, 'G_x') or not warm_start:
            self.model.precompute_gram()

        tmp = self.model.G_xt + self.lbda_reg * \
            n * m * m * torch.eye(n * m * m)
        tmp = torch.inverse(tmp)
        alpha_0 = proj(tmp @ y_train.reshape(-1, nf),
                       self.lbda_reg * n * m * m * kappa)

        losses = []

        for i in range(n_iter):
            alpha_0 -= stepsize * self.grad_huber(alpha_0)
            alpha_0 = proj(alpha_0, self.lbda_reg * n * m * m * kappa)
            losses.append(self.huber_loss_reg(alpha_0))

        self.model.alpha = alpha_0.reshape(n * m, m, nf)

        return(losses)

    def grad_epsilon(self, alpha):
        A = self.lbda_reg * self.model.n * self.model.m * alpha
        B = self.model.G_xt @ alpha @ self.model.kernel_freq
        C = self.model.y_train.reshape(self.model.n * self.model.m, -1)
        return A + B - C

    def obj_eps_dual(self, alpha):
        A = 0.5 * torch.trace(self.lbda_reg * self.model.n *
                              self.model.m * alpha @ self.model.kernel_freq @ alpha.T)
        B = 0.5 * torch.trace(self.model.G_xt @ alpha @
                              self.model.kernel_freq @ alpha.T)
        C = torch.trace(alpha @ self.model.y_train.reshape(
            self.model.n * self.model.m, -1).T)
        D = self.epsilon * torch.sqrt((alpha**2).sum(1)).sum(0)
        return A + B - C + D

    def fit_epsilon(self, data, epsilon, stepsize, n_iter=50, warm_start=False):

        n, m, nf = data.shape
        x_train = data.reshape(-1, nf)
        y_train = torch.zeros(m * n, m, nf)
        for i in range(m * n):
            y_train[i] = data[i // m]

        self.model.n = n * m
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.x_train = x_train
        self.model.y_train = y_train
        if not hasattr(self.model, 'G_x') or not warm_start:
            self.model.precompute_gram()

        tmp = self.model.G_xt + self.lbda_reg * \
            n * m * m * torch.eye(n * m * m)
        tmp = torch.inverse(tmp)
        alpha_0 = tmp @ y_train.reshape(-1, nf)

        self.epsilon = epsilon
        losses = []

        for i in range(n_iter):
            alpha_0 -= stepsize * self.grad_epsilon(alpha_0)
            alpha_0 = bst(alpha_0, epsilon)
            losses.append(self.obj_eps_dual(alpha_0))

            if i > 0:
                delta = torch.abs((losses[-2] - losses[-1]) / losses[-1])
                stop_cond = (delta < 1e-5)
                if stop_cond:
                    break

        self.model.alpha = alpha_0.reshape(n * m, m, nf)

        return(losses)

    def fit_epsilon_l1(self, data, epsilon, stepsize, n_iter=50, warm_start=False):

        n, m, nf = data.shape
        x_train = data.reshape(-1, nf)
        y_train = torch.zeros(m * n, m, nf)
        for i in range(m * n):
            y_train[i] = data[i // m]

        self.model.n = n * m
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.x_train = x_train
        self.model.y_train = y_train
        if not hasattr(self.model, 'G_x') or not warm_start:
            self.model.precompute_gram()

        tmp = self.model.G_xt + self.lbda_reg * \
            n * m * m * torch.eye(n * m * m)
        tmp = torch.inverse(tmp)
        alpha_0 = tmp @ y_train.reshape(-1, nf)

        self.epsilon = epsilon
        losses = []

        for i in range(n_iter):
            alpha_0 -= stepsize * self.grad_epsilon(alpha_0)
            alpha_0 = iht(alpha_0, epsilon)
            losses.append(self.obj_eps_dual(alpha_0))

            if i > 0:
                delta = torch.abs((losses[-2] - losses[-1]) / losses[-1])
                stop_cond = (delta < 1e-5)
                if stop_cond:
                    break

        self.model.alpha = alpha_0.reshape(n * m, m, nf)

        return(losses)

    def fit_dim_red_closed_form(self, data, warm_start=False):
        n, m, nf = data.shape
        x_train = data.reshape(-1, nf)
        y_train = torch.zeros(m * n, m, nf)
        for i in range(m * n):
            y_train[i] = data[i // m]

        self.model.n = n * m
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.x_train = x_train
        self.model.y_train = y_train
        if not hasattr(self.model, 'G_x') or not warm_start:
            self.model.precompute_gram()
        tmp = self.model.G_xt + self.lbda_reg * \
            n * m * m * torch.eye(n * m * m)
        tmp = torch.inverse(tmp)

        def proj(s, d):
            return torch.diag_embed(torch.Tensor([1 for i in range(s)] + [0 for i in range(d - s)]))[:, :s]

        # alpha, _ = torch.solve(y_train.reshape(-1,nf) @ self.model.v @ proj(self.model.s, self.model.num_freq) , tmp)
        # self.model.alpha= alpha.reshape(n*m, m, self.model.s)

        self.model.alpha = (tmp @ y_train.reshape(-1, nf) @ self.model.v @ proj(
            self.model.s, self.model.num_freq)).reshape(n * m, m, self.model.s)

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
            loss.backward(retain_graph=True)
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


class ITLEstimatorJointPartial(object):
    "ITL Class with fitting procedure using pytorch"

    def __init__(self, model, cost, lbda_reg, lbda_dkl, sampler, mask):
        self.model = model
        self.cost = cost
        self.lbda_reg = lbda_reg
        self.lbda_dkl = lbda_dkl
        self.sampler = sampler
        self.mask = mask

    def objective(self, x, y, thetas, mask):
        "Computes the objectif function to be minimized, sum of the cost +regularization"
        pred = self.model.forward(x, thetas)
        obj = self.cost(y, pred, thetas, mask).mean()
        obj += 0.5 * self.lbda_dkl * self.model.vv_norm()
        if self.model.kernel_input.is_learnable:
            obj += self.model.kernel_input.regularization()
        if self.model.kernel_output.is_learnable:
            obj += self.model.kernel_output.regularization()
        return(obj)

    def fast_objective(self):
        "Computes the objectif function to be minimized, sum of the cost +regularization"
        pred = self.model.fast_forward()
        obj = self.cost(self.model.y_train, pred,
                        self.model.thetas, self.output_mask)
        obj += 0.5 * self.lbda_reg * self.model.fast_vv_norm()
        return(obj)

    def training_risk(self):
        if not hasattr(self, 'output_mask'):
            raise ValueError('No data given')

        pred = self.model.forward(self.model.x_train, self.model.thetas)
        return self.cost(self.model.y_train, pred, self.model.thetas, self.output_mask)

    def risk(self, data, mask=None):
        n, m, nf = data.shape
        x = data.reshape(n * m, nf)
        y = torch.zeros(n * m, m, nf)
        for i in range(n * m):
            y[i] = data[i // m]

        pred = self.model.forward(x, self.model.thetas)
        return self.cost(y, pred, self.model.thetas, mask)

    def fit_closed_form(self, data):
        """Fits using a closed form solution - involves inverting several matrices"""

        x_train = data[self.mask]
        n, nf = x_train.shape
        m = data.shape[1]
        y_train = torch.zeros(n, m, nf)
        output_mask = torch.zeros((n, m), dtype=torch.bool)
        count = 0
        for i in range(data.shape[0]):
            for j in range(m):
                if self.mask[i, j].item():
                    y_train[count] = data[i]
                    output_mask[count] = self.mask[i]
                    count += 1

        self.model.n = n
        self.model.m = m
        self.sampler.m = m
        self.model.thetas = self.sampler.sample(self.model.m)

        self.model.x_train = x_train
        self.model.y_train = y_train
        self.output_mask = output_mask

        self.model.precompute_gram()

        tmp = self.model.G_xt + self.lbda_reg * n * m * torch.eye(n * m)
        mask_ravel = output_mask.reshape(-1)
        tmp = tmp[mask_ravel][:, mask_ravel]
        tmp = torch.inverse(tmp)

        #alpha_sol, _ = torch.solve(y_train[output_mask], tmp)

        alpha = torch.zeros(n, m, nf)
        #alpha[output_mask] = alpha_sol
        alpha[output_mask] = tmp @ y_train[output_mask]

        self.model.alpha = alpha

        #print('alpha fitted, empirical risk=', self.training_risk())

    def fit_kernel_input(self, data, batch_size=0.8, data_test=None, n_epochs=20, solver='sgd'):

        optimizer_kernel = optim.SGD(
            self.model.kernel_input.model.parameters(),
            lr=self.model.kernel_input.optim_params['lr'],
            momentum=self.model.kernel_input.optim_params['momentum'],
            dampening=self.model.kernel_input.optim_params['dampening'],
            weight_decay=self.model.kernel_input.optim_params['weight_decay'],
            nesterov=self.model.kernel_input.optim_params['nesterov'])

        def closure_kernel():
            loss = self.fast_objective()
            optimizer_kernel.zero_grad()
            loss.backward(retain_graph=True)
            return loss

        if not hasattr(self.model.kernel_input, 'losses'):
            self.model.kernel_input.losses = []
            self.model.kernel_input.times = [0]

        n, m, _ = data.shape

        if data_test is not None:
            self.test_risk = []

        t0 = time.time()

        for k in range(n_epochs):
            self.mask = (torch.randperm(n * m).reshape(n, m)
                         <= batch_size * n * m)
            self.fit_closed_form(data)
            loss = closure_kernel()
            optimizer_kernel.step(closure_kernel)
            self.model.kernel_input.losses.append(loss.item())
            self.model.kernel_input.times.append(time.time() - t0)
            if data_test is not None:
                self.test_risk.append(self.risk(data_test))

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
