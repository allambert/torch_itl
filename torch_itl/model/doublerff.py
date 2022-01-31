"""Implement the double RFF model."""
import torch


class DoubleRFF(object):
    r"""Implement a decomposable OVK: \tilde{k}_{X} \tilde{k}_{\Theta}.

    Both \tilde{k}_{X} and \tilde{k}_{\Theta} are Random Fourier Features
    approximations of some kernel.
    """

    def __init__(self, kernel_input, kernel_output):
        r"""Initialize the kernel.

        Parameters
        ----------
        kernel_input:  torch_itl.Kernel
            Input RFF kernel \tilde{k}_{X}
        kernel_output:  torch_itl.Kernel
            Output RFF kernel \tilde{k}_{\Theta}
        Returns
        -------
        nothing
        """
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output

    def forward(self, x, thetas):
        r"""Compute the prediction of the model on data (x, thetas).

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input tensor of data
        thetas: torch.Tensor of shape (n_anchors, n_features_2)
            Locations in the $\Theta$ space
        Returns
        -------
        pred: torch.Tensor of shape (n_samples, n_anchors)
            prediction of the model
        """
        phi_x = self.kernel_input.feature_map(x)
        phi_theta = self.kernel_output.feature_map(thetas)

        pred = (phi_x @ self.alpha @ phi_theta.T)
        return pred

    def partial_derivative(self, x, thetas):
        r"""Compute the partial derivative of the model w.r.t theta.

        Parameters
        ----------
        x:  torch.Tensor of shape (n_samples, n_features_1)
            Input tensor of data
        thetas:  torch.Tensor of shape (n_anchors, n_features_2)
            Locations in the $\Theta$ space
        Returns
        -------
        partial:  torch.Tensor of shape (n_samples, n_anchors)
            Input tensor of data
        """
        phi_x = self.kernel_input.feature_map(x)
        d_phi_theta = self.kernel_output.d_feature_map(thetas)
        partial = (phi_x @ self.alpha @ d_phi_theta.T)
        return(partial)

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
            the vv-rkhs norm of the model
        """
        res = torch.trace(self.alpha @ self.alpha.T)
        return res

    def initialize(self, warm_start, requires_grad=False):
        """Initialize the parameters alpha.

        Parameters
        ----------
        warm_start: torch.bool
            Keeps previous alpha if true
        requires_grad: torch.bool
            Use if solver is based on autodiff later on
        Returns
        -------
        None
        """
        if not hasattr(self, 'alpha') or not warm_start:
            self.alpha = torch.randn(
                (2*self.kernel_input.dim_rff, 2*self.kernel_output.dim_rff),
                requires_grad)
