import torch

dtype = torch.float
device = torch.device("cpu")


def rbf_kernel(X, Y=None, gamma=None):
    """Compute rbf Gram matrix between X and Y (or X)
    Parameters
    ----------
    X: torch.Tensor of shape (n_samples_1, n_features)
       First input on which Gram matrix is computed
    Y: torch.Tensor of shape (n_samples_2, n_features), default None
       Second input on which Gram matrix is computed. X is reused if None
    gamma: float
           Gamma parameter of the kernel (see sklearn implementation)
    Returns
    -------
    K: torch.Tensor of shape (n_samples_1, n_samples_2)
       Gram matrix on X/Y
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    K_tmp = X_norm + Y_norm - 2. * torch.mm(X, torch.t(Y))
    K_tmp *= -gamma
    K = torch.exp(K_tmp)

    return K


def get_anchors_gaussian_rff(dim_input, dim_rff, gamma):
    return gamma * torch.randn(dim_input, dim_rff)


class Kernel(object):

    def __init__(self):
        pass


class Gaussian(Kernel):

    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        return rbf_kernel(X, Y, self.gamma)


class LearnableGaussian(Kernel):

    def __init__(self, gamma, model, optim_params):
        self.gamma = gamma
        self.is_learnable = True
        self.model = model
        self.optim_params = optim_params

    def compute_gram(self, X, Y=None):
        if Y is None:
            return rbf_kernel(self.model.forward(X),Y=None,gamma=self.gamma)
        else:
            return rbf_kernel(self.model.forward(X), self.model.forward(Y),self.gamma)

    def regularization(self):
        return 0

    def clear_memory(self):
        self.losses, self.times = [], [0]


class GaussianRFF(Kernel):

    def __init__(self, dim_input, dim_rff, gamma):
        self.dim_rff = dim_rff
        self.is_learnable = False
        self.anchors = get_anchors_gaussian_rff(
            dim_input, dim_rff, gamma)

    def feature_map(self, X):
        a = torch.cos(X @ self.anchors)
        b = torch.sin(X @ self.anchors)
        return 1 / torch.sqrt(torch.Tensor([self.dim_rff])) * torch.cat((a, b), 1)


class LearnableGaussianRFF(Kernel):

    def __init__(self, gamma, model, dim_model_output, dim_rff, optim_params):
        self.dim_rff = dim_rff
        self.model = model
        self.is_learnable = True
        self.anchors = get_anchors_gaussian_rff(
            dim_model_output, dim_rff, gamma)
        self.optim_params = optim_params

    def feature_map(self, X):
        a = torch.cos(self.model.forward(X) @ self.anchors)
        b = torch.sin(self.model.forward(X) @ self.anchors)
        return 1 / torch.sqrt(torch.Tensor([self.dim_rff])) * torch.cat((a, b), 1)

    def regularization(self):
        return 0

    def clear_memory(self):
        self.losses, self.times = [], [0]

class Linear(Kernel):

    def __init__(self):
        pass

    def compute_gram(self, X, Y= None):
        if Y is None:
            Y = X
        return X @ Y.T

class LearnableLinear(Kernel):

    def __init__(self, model):
        self.model = model

    def compute_gram(self, X, Y):
        if Y is None:
            Y= X
        return self.model.forward(X) @ self.model.forward(Y).T

    def clear_memory(self):
        self.losses, self.times = [], [0]
