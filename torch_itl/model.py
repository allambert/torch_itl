import torch
import torch.optim as optim
import time
from .cost import *
from .kernel import *
from .sampler import *

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

class RFFModel(Model):

    def __init__(self, kernel_input, kernel_output):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output

    def forward(self, x, thetas):
        "Computes the output of the model at point (x,thetas)"
        if not hasattr(self, 'alpha'):
            self.alpha = torch.randn((self.kernel_input.dim_rff, self.kernel_output.dim_rff), requires_grad=True)
