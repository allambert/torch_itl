"""Implement various decomposable models in the vv-RKHS family."""
from scipy.interpolate import interp1d
import torch
from .utils import kron
from abc import ABC, abstractmethod


class Decomposable(object):
    r"""Implement a decomposable OVK.

    The associated kernel is : K = k_{\mathcal{X}} k_{\Theta} A with A psd.
    """

    def __init__(self, kernel_input, kernel_output, A):
        r"""Initialize the kernel.

        Parameters
        ----------
        kernel_input:  torch_itl.Kernel
            Input kernel k_{\mathcal{X}}
        kernel_output:  torch_itl.Kernel
            Output kernel k_{\Theta}
        A:  torch.Tensor of shape (output_dim, output_dim)
            Input kernel k_{\mathcal{X}}
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
            Input kernel k_{\mathcal{X}}
        kernel_output:  torch_itl.Kernel
            Output kernel k_{\Theta}
        d:  int
            Dimension of the output
        Returns
        -------
        nothing
        """
        super().__init__(kernel_input, kernel_output, torch.eye(d))


class DecomposableIdentityScalar(object):
    """Implement a decomposable kernel with dim 1 outputs."""

    def __init__(self, kernel_input, kernel_output):
        r"""Initialize the kernel.

        Parameters
        ----------
        kernel_input:  torch_itl.Kernel
            Input kernel k_{\mathcal{X}}
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
    r"""Implement a decomposable OVK: K = k_{\mathcal{X}} T_{k_{\Theta}}.

    T_{k_\Theta} is the integral operator associated to the kernel
    T_{k_\Theta}.
    The class is an abstract class as in practice some level of approximation
    is needed.
    """

    @abstractmethod
    def compute_gram(self, x, thetas=None):
        """Abstract method for gram matrices."""
        pass

    @abstractmethod
    def compute_gram_train(self):
        """Abstract method for training gram matrices."""
        pass

    @abstractmethod
    def forward(self, x, thetas):
        """Abstract method for forward pass."""

    @abstractmethod
    def initialize(self, x, warm_start=True, requires_grad=False):
        """Abstract method for initialization."""
        pass

    @abstractmethod
    def vv_norm(self, cpt_gram=True):
        """Abstract method for vv-norm."""
        pass

    @abstractmethod
    def squared_norm_dual_variable(self):
        """Abstract method for squared norm of dual variables."""
        pass

    @abstractmethod
    def scalar_product_dual_variables_data(self):
        """Abstract method for scalar products of dual variables and data."""
        pass

    @abstractmethod
    def regularization(self):
        """Abstract method for regularization."""
        pass

    @abstractmethod
    def gradient_squared_norm_dual_variable(self):
        """Abstract method for gradient of squared norm of dual variables."""
        pass

    @abstractmethod
    def gradient_scalar_product_dual_variables_data(self):
        """Abstract method for gradient of scalar products of dual and data."""
        pass

    @abstractmethod
    def gradient_regularization(self):
        """Abstract method for gradient of regularization."""
        pass


class DecomposableIntOpEigen(DecomposableIntOp):
    r"""Implement a decomposable OVK: K = k_{\mathcal{X}} T_{k_{\Theta}}.

    T_{k_\Theta} is the integral operator associated to the kernel
    T_{k_\Theta}.
    The class is based on some eigen-based approximation of T_{k_{\Theta}}.
    """

    def __init__(self, kernel_input, kernel_output, n_eigen):
        r"""Initialize the kernel.

        Parameters
        ----------
        kernel_input:  torch_itl.Kernel
            Input kernel k_{\mathcal{X}}
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

    def compute_gram(self, x):
        """Compute the gram matrix of the model.

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input vector of samples
        Returns
        -------
        G_x: torch.Tensor of shape (n_samples , n_samples)
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

    @abstractmethod
    def forward(self, x, thetas):
        """Abstract method for forward pass."""

    @abstractmethod
    def initialize(self, x, warm_start=True, requires_grad=False):
        """Abstract method for initialization."""
        pass

    @abstractmethod
    def vv_norm(self, cpt_gram=True):
        """Abstract method for vv-norm."""
        pass

    @abstractmethod
    def squared_norm_dual_variable(self):
        """Abstract method for squared norm of dual variables."""
        pass

    @abstractmethod
    def scalar_product_dual_variables_data(self):
        """Abstract method for scalar products of dual variables and data."""
        pass

    @abstractmethod
    def regularization(self):
        """Abstract method for regularization."""
        pass

    @abstractmethod
    def gradient_squared_norm_dual_variable(self):
        """Abstract method for gradient of squared norm of dual variables."""
        pass

    @abstractmethod
    def gradient_scalar_product_dual_variables_data(self):
        """Abstract method for gradient of scalar products of dual and data."""
        pass

    @abstractmethod
    def gradient_regularization(self):
        """Abstract method for gradient of regularization."""
        pass


class DecomposableIntOpSplines(DecomposableIntOp):
    r"""Implement a decomposable OVK: K = k_{\mathcal{X}} T_{k_{\Theta}}.

    T_{k_\Theta} is the integral operator associated to the kernel
    T_{k_\Theta}.
    The class is based on a splines approximation of the dual variables.
    """

    def compute_gram(self, x, thetas=None):
        """Abstract method for gram matrices."""
        pass

    def compute_gram_train(self):
        """Abstract method for training gram matrices."""
        pass

    @abstractmethod
    def forward(self, x, thetas):
        """Abstract method for forward pass."""

    @abstractmethod
    def initialize(self, x, warm_start=True, requires_grad=False):
        """Abstract method for initialization."""
        pass

    @abstractmethod
    def vv_norm(self, cpt_gram=True):
        """Abstract method for vv-norm."""
        pass

    @abstractmethod
    def squared_norm_dual_variable(self):
        """Abstract method for squared norm of dual variables."""
        pass

    @abstractmethod
    def scalar_product_dual_variables_data(self):
        """Abstract method for scalar products of dual variables and data."""
        pass

    @abstractmethod
    def regularization(self):
        """Abstract method for regularization."""
        pass

    @abstractmethod
    def gradient_squared_norm_dual_variable(self):
        """Abstract method for gradient of squared norm of dual variables."""
        pass

    @abstractmethod
    def gradient_scalar_product_dual_variables_data(self):
        """Abstract method for gradient of scalar products of dual and data."""
        pass

    @abstractmethod
    def gradient_regularization(self):
        """Abstract method for gradient of regularization."""
        pass
