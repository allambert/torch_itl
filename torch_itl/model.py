import torch
# from .kernel import *

dtype = torch.float

def kron(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
        [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))

def proj(s, d):
    return torch.diag_embed(torch.Tensor([1 for i in range(s)]+[0 for i in range(d-s)]))[:,:s]

def identity(s, d):
    return torch.diag_embed(torch.Tensor([1 for i in range(s)]+[0 for i in range(d-s)]))

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

    def initialise(self, x_train, warm_start):
        self.x_train = x_train
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.m), requires_grad=True)


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


class SpeechSynthesisKernelModel(Model):

    def __init__(self, kernel_input, kernel_output, kernel_freq=None):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.kernel_freq = kernel_freq
        self.num_freq = self.kernel_freq.shape[0]

    def forward(self, x, thetas):
        """
        Computes the output of the model at point (x,thetas)
        Parameters
        ----------
        x
        thetas

        Returns
        -------
        output of the model at point (x,thetas)
        """
        if not hasattr(self, 'x_train'):
            raise ValueError('Model not fed with data')
        m1 = thetas.shape[0]
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(
            thetas, self.thetas)
        G_xt = self.kronecker_product(G_x, G_t)
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return (self.kernel_freq @ alpha_reshape @ G_xt.T).T.reshape((-1, m1, self.num_freq))

    def vv_norm(self):
        """
        Computes the vv-RKHS norm of the model
        Returns
        -------
        vv-RKHS norm
        """

        G_x = self.kernel_input.compute_gram(self.x_train)
        G_t = self.kernel_output.compute_gram(self.thetas)
        G_xt = self.kronecker_product(G_x, G_t)
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return torch.trace(self.kernel_freq @ alpha_reshape @ G_xt @ alpha_reshape.T)

    def precompute_gram(self):
        self.G_x = self.kernel_input.compute_gram(self.x_train)
        self.G_t = self.kernel_output.compute_gram(self.thetas)
        self.G_xt = self.kronecker_product(self.G_x, self.G_t)

    def precompute_gram(self):
        self.G_x = self.kernel_input.compute_gram(self.x_train)
        self.G_t = self.kernel_output.compute_gram(self.thetas)
        self.G_xt = self.kronecker_product(self.G_x, self.G_t)

    def feed(self, x_train, thetas):
        "Set some x_train and thetas to the model, without optimizing"
        self.n = x_train.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x_train
        self.thetas = thetas
        self.alpha = torch.randn((self.n, self.m, self.num_freq), requires_grad=True)

    def initialise(self, x_train, warm_start):
        self.x_train = x_train
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.m, self.num_freq), requires_grad=True)

    def test_mode(self, x_train, thetas, alpha):
        self.n = x_train.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x_train
        self.thetas = thetas
        self.alpha = alpha

    @staticmethod
    def kronecker_product(matrix1, matrix2):
        return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
            [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))

class JointLandmarksSynthesisKernelModel(Model):

    def __init__(self, kernel_input, kernel_output, kernel_freq=None):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.kernel_freq = kernel_freq
        self.num_freq = self.kernel_freq.shape[0]

    def forward(self, x, thetas):
        """
        Computes the output of the model at point (x,thetas)
        Parameters
        ----------
        x
        thetas

        Returns
        -------
        output of the model at point (x,thetas)
        """
        if not hasattr(self, 'x_train'):
            raise ValueError('Model not fed with data')
        m1 = thetas.shape[0]
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(
            thetas, self.thetas)
        G_xt = self.kronecker_product(G_x, G_t)
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return (self.kernel_freq @ alpha_reshape @ G_xt.T).T.reshape((-1, m1, self.num_freq))

    def fast_forward(self):
        """
        Fast Computes the output of the model at point (x_train,thetas)
        Parameters
        ----------
        Returns
        -------
        output of the model at point (x_train,thetas)
        """
        if not hasattr(self, 'G_xt'):
            raise ValueError('Gram matrices not precomputed')
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return (self.kernel_freq @ alpha_reshape @ self.G_xt.T).T.reshape((-1, self.m, self.num_freq))

    def precompute_gram(self):
        self.G_x = self.kernel_input.compute_gram(self.x_train)
        self.G_t = self.kernel_output.compute_gram(self.thetas)
        self.G_xt = self.kronecker_product(self.G_x, self.G_t)

    def vv_norm(self):
        """
        Computes the vv-RKHS norm of the model
        Returns
        -------
        vv-RKHS norm
        """

        G_x = self.kernel_input.compute_gram(self.x_train)
        G_t = self.kernel_output.compute_gram(self.thetas)
        G_xt = self.kronecker_product(G_x, G_t)
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return torch.trace(self.kernel_freq @ alpha_reshape @ G_xt @ alpha_reshape.T)

    def fast_vv_norm(self):
        if not hasattr(self, 'G_xt'):
            raise ValueError('Gram matrices not precomputed')
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return torch.trace(self.kernel_freq @ alpha_reshape @ self.G_xt @ alpha_reshape.T)

    def identity_regularization(self):
        """
        Computes the regularization associated to enforcing identity on one subject of the exxperience
        Returns
        -------
        real
        """
        G_x = self.kernel_input.compute_gram(self.x_train)
        pass

    def feed(self, x_train, thetas):
        "Set some x_train and thetas to the model, without optimizing"
        self.n = x_train.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x_train
        self.thetas = thetas
        self.alpha = torch.randn((self.n, self.m, self.num_freq), requires_grad=True)

    def initialise(self, x_train, y_train, warm_start):
        self.x_train = x_train
        self.y_train = y_train
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.m, self.num_freq), requires_grad=True)

    def test_mode(self, x_train, thetas, alpha):
        self.n = x_train.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x_train
        self.thetas = thetas
        self.alpha = alpha

    @staticmethod
    def kronecker_product(matrix1, matrix2):
        return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
            [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))


class LandmarksSynthesisDimReductionKernelModel(Model):

    def __init__(self, kernel_input, kernel_output, s, v, nf):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.s = s
        self.v = v
        self.num_freq = nf
        self.kernel_freq = v @ identity(s, nf) @ v.T

    def forward(self, x, thetas):
        """
        Computes the output of the model at point (x,thetas)
        Parameters
        ----------
        x
        thetas

        Returns
        -------
        output of the model at point (x,thetas)
        """
        if not hasattr(self, 'x_train'):
            raise ValueError('Model not fed with data')
        m1 = thetas.shape[0]
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(
            thetas, self.thetas)
        G_xt = self.kronecker_product(G_x, G_t)
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.s))
        return (G_xt @ alpha_reshape @ proj(self.s, self.num_freq).T @ self.v.T).reshape((-1, m1, self.num_freq))

    def fast_forward(self):
        """
        Fast Computes the output of the model at point (x_train,thetas)
        Parameters
        ----------
        Returns
        -------
        output of the model at point (x_train,thetas)
        """
        if not hasattr(self, 'G_xt'):
            raise ValueError('Gram matrices not precomputed')
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return (self.kernel_freq @ alpha_reshape @ self.G_xt.T).T.reshape((-1, self.m, self.num_freq))

    def precompute_gram(self):
        self.G_x = self.kernel_input.compute_gram(self.x_train)
        self.G_t = self.kernel_output.compute_gram(self.thetas)
        self.G_xt = self.kronecker_product(self.G_x, self.G_t)

    def vv_norm(self):
        """
        Computes the vv-RKHS norm of the model
        Returns
        -------
        vv-RKHS norm
        """

        G_x = self.kernel_input.compute_gram(self.x_train)
        G_t = self.kernel_output.compute_gram(self.thetas)
        G_xt = self.kronecker_product(G_x, G_t)
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return torch.trace(self.kernel_freq @ alpha_reshape @ G_xt @ alpha_reshape.T)

    def fast_vv_norm(self):
        if not hasattr(self, 'G_xt'):
            raise ValueError('Gram matrices not precomputed')
        alpha_reshape = torch.reshape(self.alpha, (self.n * self.m, self.num_freq)).T
        return torch.trace(self.kernel_freq @ alpha_reshape @ self.G_xt @ alpha_reshape.T)

    def identity_regularization(self):
        """
        Computes the regularization associated to enforcing identity on one subject of the exxperience
        Returns
        -------
        real
        """
        G_x = self.kernel_input.compute_gram(self.x_train)
        pass

    def feed(self, x_train, thetas):
        "Set some x_train and thetas to the model, without optimizing"
        self.n = x_train.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x_train
        self.thetas = thetas
        self.alpha = torch.randn((self.n, self.m, self.num_freq), requires_grad=True)

    def initialise(self, x_train, y_train, warm_start):
        self.x_train = x_train
        self.y_train = y_train
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.m, self.num_freq), requires_grad=True)

    def test_mode(self, x_train, thetas, alpha):
        self.n = x_train.shape[0]
        self.m = thetas.shape[0]
        self.x_train = x_train
        self.thetas = thetas
        self.alpha = alpha

    @staticmethod
    def kronecker_product(matrix1, matrix2):
        return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
            [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))
