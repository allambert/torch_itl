from .vitl import VITL
from .utils import ploss_with_crossing


class IQR(VITL):
    """Implements Infinite Quantile Regression as proposed in
    'Infinite Task Learning in RKHSs'
    """

    def __init__(self, model, lbda, lbda_cross, sampler):
        super().__init__(model, ploss_with_crossing(lbda_cross), lbda, sampler)

    def fit_alpha_sgd(self, x, y, n_epochs=500, warm_start=True, **kwargs):
        """
        Fits the parameters alpha of the model, based on the representer t
        heorem with gradient descent

        Parameters
        ----------
        x: torch.Tensor, shape (n_samples, n_features_1)
            Input vector of samples used in the empirical risk
        y: torch.Tensor, shape (n_samples, n_features_2)
            Output vector of samples used in the empirical risk
        n_epochs: int
            Max number of iterations for training
        solver: torch.Optimizer
            Prefered gradient descent algorithm -- beware to match **kwargs
        warm_start: bool
            Keep previous estimate of alpha (or not)

        Returns
        -------
        Nothing
        """
        n = x.shape[0]
        self.model.initialize(x, warm_start)
