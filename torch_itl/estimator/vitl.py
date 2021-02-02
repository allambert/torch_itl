import torch


class VITL(object):
    """Implements VITL
    It is a vectorial outputs extension to ITL, proposed in
    'Infinite Task Learning in RKHSs'
    """

    def __init__(self, model, cost, lbda, sampler):
        self.model = model
        self.cost = cost
        self.lbda = lbda
        self.sampler = sampler

    def predict(self, x, thetas=None):
        """
        Computes the prediction of the estimator on specific data
        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, n_features_1)
           Input vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_samples, n_anchors, n_features_3)
           Locations associated to the sampled empirical risk
           default: locations used when learning
        Returns
        -------
        pred: torch.Tensor of shape (n_samples, n_anchors, n_features_2)
           prediction of the model
        """
        if thetas is None:
            if not hasattr(self.model, 'thetas'):
                raise Exception('No thetas provided')
            else:
                thetas = self.model.thetas
        return self.model.forward(x, thetas)

    def objective(self, x, y, thetas):
        """
        Computes the objective function associated to the
        regularized vITL problem
        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, n_features_1)
           Input vector of samples used in the empirical risk
        y: torch.Tensor of shape (n_samples, n_features_2)
           Output vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_samples, n_anchors, n_features_3)
           Locations associated to the sampled empirical risk
        Returns
        -------
        obj: torch.Tensor of shape (1)
           Value of the objective function
        """
        pred = self.model.forward(x, thetas)
        obj = self.cost(y, pred, thetas)
        obj += 0.5 * self.lbda * self.model.vv_norm()
        if self.model.kernel_input.is_learnable:
            obj += self.model.kernel_input.regularization()
        if self.model.kernel_output.is_learnable:
            obj += self.model.kernel_output.regularization()
        return obj

    def risk(self, x, y, thetas=None):
        """
        Computes the empirical risk of the estimator on specific data
        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        y: torch.Tensor of shape (n_samples, n_features_2)
            Output vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_samples, n_anchors, n_features_3)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        Returns
        -------
        obj: torch.Tensor of shape (1)
            Value of the empirical risk
        """
        if thetas is None:
            if not hasattr(self.model, 'thetas'):
                raise Exception('No thetas provided')
            else:
                thetas = self.model.thetas
        pred = self.model.forward(x, thetas)
        return self.cost(y, pred, thetas)

    def training_risk(self):
        """
        Computes the empirical risk of the estimator on training data
        Parameters
        ----------
        None
        Returns
        -------
        obj: torch.Tensor of shape (1)
           Value of the training empirical risk
        """
        if not hasattr(self.model, 'x_train'):
            raise Exception('No training data provided')
        return self.risk(self.model.x_train, self.model.y_train)

    def fit_alpha_gd(self, x, y, n_epochs=500, solver=torch.optim.LBFGS, warm_start=True, **kwargs):
        """
        Fits the parameters alpha of the model, based on the representer theorem
        with gradient descent
        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        y: torch.Tensor of shape (n_samples, n_features_2)
            Output vector of samples used in the empirical risk
        n_epochs: Int
            Max number of iterations for training
        solver: torch.Optimizer
            Prefered gradient descent algorithm -- beware to match **kwargs
        warm_start: Bool
            Keep previous estimate of alpha (or not)
        **kwargs:
            Arguments to be used with the optimizer
        Returns
        -------
        Nothing
        """
        # To modify: initialization inside a different function
        n = x.shape[0]
        if not hasattr(self.sampler, 'm'):
            self.model.m = int(torch.floor(
                torch.sqrt(torch.Tensor([n]))).item())
            self.sampler.m = self.model.m
        else:
            self.model.m = self.sampler.m
        self.model.thetas = self.sampler.sample(self.model.m)
        self.model.n = n

        self.model.initialise(x, warm_start)

        if not hasattr(self, 'losses'):
            self.losses = []
            self.times = [0]

        optimizer_alpha = solver([self.model.alpha], **kwargs)

        def closure_alpha():
            loss = self.objective(x, y, self.model.thetas)
            optimizer_alpha.zero_grad()
            loss.backward()
            return(loss)

        t0 = time.time()

        for k in range(n_epochs):
            loss = closure_alpha()
            self.losses.append(loss.item())
            self.times.append(time.time() - t0)
            optimizer_alpha.step(closure_alpha)

    def fit_kernel_input(self, x, y, n_epochs=150):
        """
        Fits the parameters of the neural network associated to the input kernel
        (deep kernel learning) with gradient descent
        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        y: torch.Tensor of shape (n_samples, n_features_2)
            Output vector of samples used in the empirical risk
        n_epochs: Int
            Max number of iterations for training
        warm_start: Bool
            Keep previous estimate of alpha (or not)
        **kwargs:
            Arguments to be used with the optimizer
        Returns
        -------
        Nothing
        """
        # put correct initializer
        optimizer_kernel = optim.SGD(
            self.model.kernel_input.model.parameters(),
            lr=self.model.kernel_input.optim_params['lr'],
            momentum=self.model.kernel_input.optim_params['momentum'],
            dampening=self.model.kernel_input.optim_params['dampening'],
            weight_decay=self.model.kernel_input.optim_params['weight_decay'],
            nesterov=self.model.kernel_input.optim_params['nesterov'])

        def closure_kernel():
            loss = self.objective(x, y, self.model.thetas)
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

    def fit_kernel_output(self, x, y, n_epochs=150):
        """
        Fits the parameters of the neural network associated to the output kernel
        (deep kernel learning)
        Parameters
        ----------
        x: torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        y: torch.Tensor of shape (n_samples, n_features_2)
            Output vector of samples used in the empirical risk
        n_epochs: Int
            Max number of iterations for training
        warm_start: Bool
            Keep previous estimate of alpha (or not)
        **kwargs:
            Arguments to be used with the optimizer
        Returns
        -------
        Nothing
        """
        # put correct initializer
        optimizer_kernel = optim.SGD(
            self.model.kernel_output.model.parameters(),
            lr=self.model.kernel_output.optim_params['lr'],
            momentum=self.model.kernel_output.optim_params['momentum'],
            dampening=self.model.kernel_output.optim_params['dampening'],
            weight_decay=self.model.kernel_output.optim_params['weight_decay'],
            nesterov=self.model.kernel_output.optim_params['nesterov'])

        def closure_kernel():
            loss = self.objective(x, y, self.model.thetas)
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
        self.losses, self.times = [], []
        if self.model.kernel_input.is_learnable:
            self.model.kernel_input.clear_memory()
        if self.model.kernel_output.is_learnable:
            self.model.kernel_output.clear_memory()
