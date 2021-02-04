import torch

from .utils import rbf_kernel, get_anchors_gaussian_rff


class Gaussian(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        return rbf_kernel(X, Y, self.gamma)


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


class Linear(object):

    def __init__(self):
        pass

    def compute_gram(self, X, Y=None):
        if Y is None:
            Y = X
        return X @ Y.T
