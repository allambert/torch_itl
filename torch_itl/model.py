import torch
# from .kernel import *

dtype = torch.float


class Model(object):

    def __init__(self):
        pass


class KernelModel(Model):

    def __init__(self, kernel_input, kernel_output):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output

    def forward(self, x, thetas):
        "Computes the output of the model at point (x,thetas)"
        if not hasattr(self, 'x_train'):
            raise ValueError('Model not feeded with data')
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(
            self.thetas, thetas)
        return 1 / self.n * G_x @ self.alpha @ G_t

    def vv_norm(self):
        "Computes the vv-RKHS norm of the model"
        G_x = self.kernel_input.compute_gram(self.x_train)
        G_t = self.kernel_output.compute_gram(self.thetas)
        return 1 / self.n**2 * torch.trace(G_x @ self.alpha @ G_t @ self.alpha.T)

    def feed(self, x_train, thetas):
        "Set some x_train and thetas to the model, without optimizing"
        self.n = x_train.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x_train
        self.thetas = thetas
        self.alpha = torch.randn((self.n, self.m), requires_grad=True)

    def initialise(self, x_train):
        self.model.x_train = x_train
        if not hasattr(self.model, 'alpha') or not warm_start:
            self.model.alpha = torch.randn(
                (self.model.n, self.model.m), requires_grad=True)


class RFFModel(Model):

    def __init__(self, kernel_input, kernel_output):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output

    def forward(self, x, thetas):
        "Computes the output of the model at point (x,thetas)"
        feature_map_input = self.kernel_input.feature_map(x)
        feature_map_output = self.kernel_output.feature_map(thetas)
        return feature_map_input @ self.alpha @ feature_map_output.T

    def vv_norm(self):
        return torch.trace(self.alpha @ self.alpha.T)

    def initialise(self, x_train, warm_start):
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (2 * self.kernel_input.dim_rff, 2 * self.kernel_output.dim_rff), requires_grad=True)
