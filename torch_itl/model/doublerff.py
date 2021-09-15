import torch


def kron(matrix1, matrix2):
    return torch.ger(
        matrix1.view(-1),
        matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
        [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0),
                              matrix1.size(1) * matrix2.size(1))


class DoubleRFF(object):
    r"""
    Implements a decomposable OVK: \tilde{k}_{X} \tilde{k}_{\Theta} where
    """

    def __init__(self, kernel_input, kernel_output):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output


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
        phi_x = self.kernel_input.feature_map(x)
        phi_theta = self.kernel_output.feature_map(thetas)

        pred = (phi_x @ self.alpha @ phi_theta.T)
        return pred

    def partial_derivative(self, x, thetas):
        """
        # TODO:
        """
        phi_x = self.kernel_input.feature_map(x)
        d_phi_theta = self.kernel_output.d_feature_map(thetas)
        pred = (phi_x @ self.alpha @ d_phi_theta.T)
        return(pred)

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
        res = torch.trace(self.alpha @ self.alpha.T)
        return res

    def initialise(self, x, warm_start):
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
                (2*self.kernel_input.dim_rff, 2*self.kernel_output.dim_rff), requires_grad=True)

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
