"""Implement various decomposable models in the vv-RKHS family."""

from abc import ABC
from scipy.interpolate import interp1d
import torch
from.utils import kron


class Decomposable(ABC):
    r"""Implement a decomposable OVK.

    The associated kernel is : K = k_{X} k_{\Theta} A with A psd.
    """

    def __init__(self, kernel_input, kernel_output, A):
        r"""Initialize the kernel.

        Parameters
        ----------
        kernel_input:  torch_itl.Kernel
            Input kernel k_{X}
        kernel_output:  torch_itl.Kernel
            Output kernel k_{\Theta}
        A:  torch.Tensor of shape (output_dim, output_dim)
            Input kernel k_{X}
        Returns
        -------
        nothing
        """
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.A = A
        self.output_dim = self.A.shape[0]

    def compute_gram(self, x, thetas):
        """Compute the gram matrices.

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
            default: locations used when learning
        Returns
        -------
        G_xt: torch.Tensor, of shape (n_samples*n_anchors, n_samples*n_anchors)
            Gram matrix used for predictions
        """
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(thetas, self.thetas)
        G_xt = kron(G_x, G_t)
        return G_xt

    def compute_gram_train(self):
        """Compute and stores the gram matrices of the training data.

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
        """Compute the prediction of the model on specific data.

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
        """Compute the vv-RKHS norm of the model.

        Based on with the representer theorem with parameters alpha.
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

    def initialize(self, x, warm_start, requires_grad=False):
        """Initialize the parameters alpha given by the representer theorem.

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

    def load(self, x, thetas, alpha):
        """Load a model with (x,thetas,alpha) as in the representer theorem.

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input tensor of training data
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations associated to the sampled empirical risk
        alpha: torch.Tensor of shape (n_samples, n_anchors, self.output_dim)
            Coefficients in the representer theorem
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
    """Implement a decomposable kernel with identity matrix."""

    def __init__(self, kernel_input, kernel_output, d):
        r"""Initialize the kernel.

        Parameters
        ----------
        kernel_input:  torch_itl.Kernel
            Input kernel k_{X}
        kernel_output:  torch_itl.Kernel
            Output kernel k_{\Theta}
        d:  int
            Dimension of the output
        Returns
        -------
        nothing
        """
        super().__init__(kernel_input, kernel_output, torch.eye(d))


class DecomposableIdentityScalar(ABC):
    """Implement a decomposable kernel with dim 1 outputs."""

    def __init__(self, kernel_input, kernel_output):
        r"""Initialize the kernel.

        Parameters
        ----------
        kernel_input:  torch_itl.Kernel
            Input kernel k_{X}
        kernel_output:  torch_itl.Kernel
            Output kernel k_{\Theta}
        Returns
        -------
        nothing
        """
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.thetas = None
        self.x_train = None
        self.n = None
        self.y_train = None
        self.thetas = None
        self.m = None

    def compute_gram(self, x, thetas):
        """Compute the gram matrices of the model anchors and (x,thetas).

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations in the sampled empirical risk
        Returns
        -------
        G_x: torch.Tensor, of shape (n_samples, n_samples)
        G_t: torch.Tensor of shape (n_anchors, n_anchors)
        """
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        G_t = self.kernel_output.compute_gram(thetas, self.thetas)
        return G_x, G_t

    def compute_gram_train(self):
        """Compute and store the gram matrices of the training data.

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if not hasattr(self, 'x_train') or not hasattr(self, 'thetas'):
            raise Exception('No training data or locations provided')
        self.G_x = self.kernel_input.compute_gram(self.x_train)
        self.G_t = self.kernel_output.compute_gram(self.thetas)

    def forward(self, x, thetas):
        """Compute the prediction of the model on specific data.

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations at which the prediction is computed
        Returns
        -------
        pred: torch.Tensor of shape (n_samples, n_anchors)
            Prediction of the model
        """
        if not hasattr(self, 'x_train'):
            raise Exception('No training anchors provided to the model')

        G_x, G_t = self.compute_gram(x, thetas)
        pred = G_x @ self.alpha @ G_t
        return pred

    def vv_norm(self, cpt_gram=True):
        """Compute the vv-RKHS norm of the model.

        Parameters
        ----------
        cpt_gram: torch.bool
            Use if you want to re-compute the gram matrix, default is True
        Returns
        -------
        res: torch.Tensor of shape (1)
            Vv-rkhs norm of the model
        """
        if cpt_gram:
            self.compute_gram_train()
        res = torch.trace(self.G_x @ self.alpha @ self.G_t @ self.alpha)
        return res

    def initialize(self, x, warm_start, requires_grad=False):
        """Initialize the parameters alpha given by the representer theorem.

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Training data to be memorized
        warm_start: torch.bool
            Keeps previous alpha if true
        Returns
        -------
        None
        """
        self.x_train = x
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.m), requires_grad=requires_grad)
        # For cross validation, the shape of the alpha
        # may vary which can cause errors
        elif warm_start and hasattr(self, 'alpha'):
            if len(self.alpha) != self.n:
                self.alpha = torch.randn((self.n, self.m),
                                         requires_grad=requires_grad)


class DecomposableIntOp(ABC):
    r"""Implement a decomposable OVK: K = k_{X} T_{k_{\Theta}}.

    T_{k_\Theta} is the integral operator associated to the kernel T_{k_\Theta}
    """

    def __init__(self, kernel_input, kernel_output, n_eigen):
        r"""Initialize the kernel.

        Parameters
        ----------
        kernel_input:  torch_itl.Kernel
            Input kernel k_{X}
        kernel_output:  torch_itl.Kernel
            Output kernel k_{\Theta}
        n_eigen:  int
            Number of eigenvalues used in approximation of T_{k_\Theta}
        Returns
        -------
        nothing
        """
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.n_eigen = n_eigen
        self.R = None
        self.eig_vals = None
        self.eig_vecs = None
        self.thetas = None
        self.x_train = None
        self.n = None
        self.y_train = None
        self.thetas = None
        self.m = None

    def compute_gram(self, x):
        """Compute the gram matrix of the model anchors and x.

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        Returns
        -------
        G_x: torch.Tensor, \
                shape (n_samples , n_samples)
            Gram matrix used for predictions
        """
        G_x = self.kernel_input.compute_gram(x, self.x_train)
        return G_x

    def compute_gram_train(self):
        """Compute and store the gram matrix of the training data.

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

    def compute_eigen_output(self):
        r"""Compute and store the approximate eigendecomposition of T_{k_\Theta}.

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        eig_vecs, eig_vals, _ = torch.linalg.svd(
            self.kernel_output.compute_gram(self.thetas))
        self.eig_vecs = eig_vecs[:, :self.n_eigen].T
        self.eig_vals = eig_vals[:self.n_eigen]

    def compute_R(self):
        r"""Compute and store the scalar products of data and eigenbasis.

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.R = self.y_train @ self.eig_vecs.T

    # def compute_R(self, thetas, y):
    #     d = thetas.shape[0]
    #     psi = self.kernel_output.compute_psi(thetas)
    #     self.R = y @ psi/d

    def forward(self, x, thetas):
        r"""Compute the prediction of the model on specific data (x, theta).

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations in the $\Theta$ space
        Returns
        -------
        pred: torch.Tensor of shape (n_samples, n_anchors)
            Prediction of the model
        """
        if not hasattr(self, 'x_train'):
            raise Exception('No training anchors provided to the model')

        G_x = self.compute_gram(x)
        pred = G_x @ self.alpha @ torch.diag(self.eig_vals) @ self.eig_vecs
        # Use linear interpolation for new thetas
        interp_func = interp1d(self.thetas[:, 0].numpy(), pred.numpy(),
                               fill_value="extrapolate")
        return torch.from_numpy(interp_func(thetas[:, 0]))
        # G_t = self.kernel_output.compute_psi(thetas)
        # n = x.shape[0]
        # m = thetas.shape[0]
        # pred = G_x @ self.alpha @ self.Lambda @ G_t.T
        # return pred

    def initialize(self, x, warm_start=True, requires_grad=False):
        """Initialize the parameters alpha given by the representer theorem.

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input tensor of training data used in the empirical risk
        warm_start: torch.bool
            Keeps previous alpha if true
        requires_grad: torch.bool
            Use if optimizer is based on autodiff
        Returns
        -------
        None
        """
        self.x_train = x
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (self.n, self.n_eigen), requires_grad=requires_grad)

    def vv_norm(self, cpt_gram=True):
        """Compute the vv-RKHS norm of the model.

        Parameters
        ----------
        cpt_gram: torch.bool
            Use if you want to compute the gram matrix, default is True
        None
        Returns
        -------
        res: torch.Tensor of shape (1)
            Vv-rkhs norm of the model
        """
        if cpt_gram:
            self.compute_gram_train()
        res = torch.trace(self.G_x @ self.alpha
                          @ torch.diag(self.eig_vals) @ self.alpha.T)
        return res
