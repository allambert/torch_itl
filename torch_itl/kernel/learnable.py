import torch
from .kernel import Gaussian, GaussianRFF, Linear


class LearnableKernel(object):

    def __init__(self, kernel, model, optim_params):
        self.kernel = kernel
        self.is_learnable = True
        self.model = model
        self.optim_params = optim_params

    def compute_gram(self, X, Y=None):
        if Y is None:
            return self.kernel.compute_gram(self.model.forward(X), Y=None)
        else:
            return self.kernel.compute_gram(self.model.forward(X), self.model.forward(Y))

    def regularization(self):
        return 0

    def clear_memory(self):
        self.losses, self.times = [], [0]

class LearnableLinear(LearnableKernel):

    def __init__(self, model, optim_params):
        super().__init__(Linear, model, optim_params)

class LearnableGaussian(LearnableKernel):

    def __init__(self, gamma, model, optim_params):
        super().__init__(Gaussian(gamma), model, optim_params)

class LearnableGaussianRFF(LearnableKernel):

    def __init__(self, gamma, model, dim_model_output, dim_rff, optim_params):
        super().__init__(GaussianRFF(dim_model_output, dim_rff, gamma), model, optim_params)

    def feature_map(self, X):
        return self.kernel.feature_map(self.model.forward(X))

    def regularization(self):
        return 0

    def clear_memory(self):
        self.losses, self.times = [], [0]
