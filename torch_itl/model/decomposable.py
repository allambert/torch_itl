import torch


def kron(matrix1, matrix2):
    return torch.ger(
        matrix1.view(-1),
        matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
        [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0),
                              matrix1.size(1) * matrix2.size(1))


class Decomposable(object):
    r"""
    Implements a decomposable OVK: k_{X} k_{\Theta} A
    where A is a positive semi definite matrix
    """

    def __init__(self, kernel_input, kernel_output, A):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.A = A
        self.output_dim = self.A.shape[0]

    def compute_gram(self, x, thetas):
        """
        Compute and store the gram matrices of the model anchors and (x,thetas)
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        Returns
        -------
        G_xt: torch.Tensor, \
                shape (n_samples * n_anchors, n_samples * n_anchors)
            Gram matrix used for predictions
        """
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(thetas, self.thetas)
        G_xt = kron(G_x, G_t)
        return G_xt

    def compute_gram_train(self):
        """
        Computes and stores the gram matrices of the training data
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if not hasattr(self, 'x_train'):
            raise Exception('No training data provided')
        self.G_x = self.kernel_input.compute_gram(self.x_train)
        self.G_t = self.kernel_output.compute_gram(self.thetas)
        self.G_xt = kron(self.G_x, self.G_t)

    def forward(self, x, thetas):
        """
        Computes the prediction of the model on specific data
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        Returns
        -------
        pred: torch.Tensor of shape (n_samples, n_anchors, self.output_dim)
            prediction of the model
        """
        if not hasattr(self, 'x_train'):
            raise Exception('No training anchors provided to the model')

        G_xt = self.compute_gram(x, thetas)
        n = x.shape[0]
        m = thetas.shape[0]
        alpha_reshape = self.alpha.reshape(self.n * self.m, self.output_dim)
        pred = (G_xt @ alpha_reshape @ self.A).reshape(n, m, self.output_dim)
        return pred

    def vv_norm(self, cpt_gram=True):
        """
        Computes the vv-RKHS norm of the model with parameters alpha
        given by the representer theorem
        Parameters
        ----------
        cpt_gram: torch.bool
            Use if you want to compute the gram matrix, default is True
        None
        Returns
        -------
        res: torch.Tensor of shape (1)
            the vv-rkhs norm of the model
        """
        if cpt_gram:
            self.compute_gram_train()
        alpha_reshape = self.alpha.reshape(self.n * self.m, self.output_dim)
        res = torch.trace(self.G_xt @ alpha_reshape @ self.A @ alpha_reshape.T)
        return res

    def initialise(self, x, warm_start, requires_grad=True):
        """
        Initializes the parameters alpha given by the representer theorem
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input tensor of training data used in the empirical risk
        warm_start: torch.bool
            keeps previous alpha if true
        Returns
        -------
        None
        """
        self.x_train = x
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.m, self.output_dim), requires_grad=requires_grad)

    def test_mode(self, x, thetas, alpha):
        """
        Loads a model with x,thetas,alpha as in the representer theorem
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input tensor of training data used in the empirical risk
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        alpha: torch.Tensor of shape (n_samples, n_features_1, self.output_dim)
        Returns
        -------
        None
        """
        self.n = x.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x
        self.thetas = thetas
        self.alpha = alpha


class DecomposableIdentity(Decomposable):

    def __init__(self, kernel_input, kernel_output, d):
        super().__init__(kernel_input, kernel_output, torch.eye(d))


class DecomposableIdentityScalar():

    def __init__(self, kernel_input, kernel_output):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output

    def compute_gram(self, x, thetas):
        """
        Compute and store the gram matrices of the model anchors and (x,thetas)
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        Returns
        -------
        G_xt: torch.Tensor, \
                shape (n_samples * n_anchors, n_samples * n_anchors)
            Gram matrix used for predictions
        """
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(thetas, self.thetas)
        G_xt = kron(G_x, G_t)
        return G_xt

    def compute_gram_train(self):
        """
        Computes and stores the gram matrices of the training data
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if not hasattr(self, 'x_train'):
            raise Exception('No training data provided')
        self.G_x = self.kernel_input.compute_gram(self.x_train)
        self.G_t = self.kernel_output.compute_gram(self.thetas)
        self.G_xt = kron(self.G_x, self.G_t)

    def forward(self, x, thetas):
        """
        Computes the prediction of the model on specific data
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        Returns
        -------
        pred: torch.Tensor of shape (n_samples, n_anchors, self.output_dim)
            prediction of the model
        """
        if not hasattr(self, 'x_train'):
            raise Exception('No training anchors provided to the model')

        G_xt = self.compute_gram(x, thetas)
        n = x.shape[0]
        m = thetas.shape[0]
        alpha_reshape = self.alpha.reshape(self.n * self.m)
        pred = (G_xt @ alpha_reshape).reshape(n, m)
        return pred

    def vv_norm(self, cpt_gram=True):
        """
        Computes the vv-RKHS norm of the model with parameters alpha
        given by the representer theorem
        Parameters
        ----------
        cpt_gram: torch.bool
            Use if you want to compute the gram matrix, default is True
        None
        Returns
        -------
        res: torch.Tensor of shape (1)
            the vv-rkhs norm of the model
        """
        if cpt_gram:
            self.compute_gram_train()
        alpha_reshape = self.alpha.reshape(self.n * self.m)
        res = torch.trace(self.G_xt @ alpha_reshape @ alpha_reshape.T)
        return res

    def initialise(self, x, warm_start, requires_grad=True):
        """
        Initializes the parameters alpha given by the representer theorem
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input tensor of training data used in the empirical risk
        warm_start: torch.bool
            keeps previous alpha if true
        Returns
        -------
        None
        """
        self.x_train = x
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.m), requires_grad=requires_grad)


class DecomposableIntOp():
    r"""
    Implements a decomposable OVK: k_{X} T_k_{\Theta}
    """
    def __init__(self, kernel_input, kernel_output, n_eigen):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.m = n_eigen
        self.Lambda = torch.diag(kernel_output.get_eigen())

    def compute_gram(self, x):
        """
        Compute and store the gram matrices of the model anchors and (x,thetas)
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        Returns
        -------
        G_x: torch.Tensor, \
                shape (n_samples , n_samples)
            Gram matrix used for predictions
        """
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        return G_x

    def compute_gram_train(self):
        """
        Computes and stores the gram matrices of the training data
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if not hasattr(self, 'x_train'):
            raise Exception('No training data provided')
        self.G_x = self.kernel_input.compute_gram(self.x_train)

    def compute_R(self, thetas, y):
        d = thetas.shape[0]
        psi = self.kernel_output.compute_psi(thetas)
        self.R = y @ psi/d

    def forward(self, x, thetas):
        """
        Computes the prediction of the model on specific data
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        Returns
        -------
        pred: torch.Tensor of shape (n_samples, n_anchors, self.output_dim)
            prediction of the model
        """
        if not hasattr(self, 'x_train'):
            raise Exception('No training anchors provided to the model')

        G_x = self.compute_gram(x)
        G_t = self.kernel_output.compute_psi(thetas)
        n = x.shape[0]
        m = thetas.shape[0]

        pred = G_x @ self.alpha @ self.Lambda @ G_t.T
        return pred

    def vv_norm(self, cpt_gram=True):
        """
        Computes the vv-RKHS norm of the model with parameters alpha
        given by the representer theorem
        Parameters
        ----------
        cpt_gram: torch.bool
            Use if you want to compute the gram matrix, default is True
        None
        Returns
        -------
        res: torch.Tensor of shape (1)
            the vv-rkhs norm of the model
        """
        if cpt_gram:
            self.compute_gram_train()
        res = torch.trace(self.G_xt @ self.alpha @ self.Lambda @ self.alpha.T)
        return res

    def initialise(self, x, warm_start=True, requires_grad=True):
        """
        Initializes the parameters alpha given by the representer theorem
        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input tensor of training data used in the empirical risk
        warm_start: torch.bool
            keeps previous alpha if true
        Returns
        -------
        None
        """
        self.x_train = x
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.m), requires_grad=requires_grad)
