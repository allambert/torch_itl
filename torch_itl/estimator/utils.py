"""Utilitary functions for the estimators."""
import torch


# Proximal operators of indicator functions of balls w.r.t. different norms
def proj_matrix_2(alpha, kappa):
    r"""Compute the row-wise projection of a matrix on the (2, kappa)-ball.

    Parameters
    ----------
    alpha:  torch.Tensor of shape (n,m)
    kappa:  float
    Returns
    -------
    alpha: torch.Tensor of shape (n,m)
        the rows of alpha are projected on the 2-ball of radius kappa
    """
    norm = torch.sqrt(torch.sum(alpha**2, axis=1))
    mask = torch.where(norm > kappa)
    alpha[mask] *= kappa / norm[mask].reshape((-1, 1))
    return alpha


def proj_matrix_inf(alpha, kappa):
    r"""Compute the projection of a matrix on the ($\infty$, kappa)-ball.

    Parameters
    ----------
    alpha:  torch.Tensor of shape (n,m)
    kappa:  float
    Returns
    -------
    alpha: torch.Tensor of shape (n,m)
        each element of alpha is projected on $[-\kappa, \kappa]$
    """
    norm = torch.abs(alpha)
    return torch.where(norm > kappa, kappa * alpha/norm, alpha)


def proj_vect_2(alpha, kappa):
    r"""Compute the projection of a vector on the (2, kappa)-ball.

    Parameters
    ----------
    alpha:  torch.Tensor of shape (n)
    kappa:  float
    Returns
    -------
    alpha: torch.Tensor of shape (n)
        the matrix alpha is projected on the 2-ball of radius kappa
    """
    norm = torch.sqrt(torch.sum(alpha**2))
    if norm > kappa:
        alpha *= kappa / norm
    return alpha


def proj_vect_inf(alpha, kappa):
    r"""Compute the projection of a vector on the ($\infty$, kappa)-ball.

    Parameters
    ----------
    alpha:  torch.Tensor of shape (n)
    kappa:  float
    Returns
    -------
    alpha: torch.Tensor of shape (n)
        the coefficients of alpha are projected on $[-\kappa, \kappa]$
    """
    norm = torch.abs(alpha)
    mask = torch.where(norm > kappa)
    alpha[mask] *= kappa / norm[mask].reshape((-1, 1))
    return alpha


def bst_matrix(alpha, tau):
    r"""Compute the block soft thresholding of a matrix.

    Parameters
    ----------
    alpha:  torch.Tensor of shape (n,m)
    tau:  float
    Returns
    -------
    alpha: torch.Tensor of shape (n,m)
    """
    norm = (alpha**2).sum(1).sqrt()
    mask_st = torch.where(norm >= tau)
    mask_ze = torch.where(norm < tau)
    alpha[mask_st] = alpha[mask_st] - alpha[mask_st] / \
        norm[mask_st].reshape((-1, 1)) * tau
    alpha[mask_ze] = 0
    return alpha


def bst_vector(alpha, tau):
    r"""Compute the block soft thresholding of a vector.

    Parameters
    ----------
    alpha:  torch.Tensor of shape (n)
    tau:  float
    Returns
    -------
    alpha: torch.Tensor of shape (n)
    """
    norm = (alpha**2).sum().sqrt()
    if norm > tau:
        alpha -= alpha/norm * tau
    else:
        alpha = 0
    return alpha


def st(alpha, tau):
    r"""Compute the soft thresholding of a matrix.

    Parameters
    ----------
    alpha:  torch.Tensor of shape (n,m)
    tau:  float
    Returns
    -------
    alpha: torch.Tensor of shape (n,m)
    """
    return torch.where(alpha.abs() - tau < 0,
                       torch.zeros_like(alpha),
                       alpha.abs() - tau) * torch.sign(alpha)


def squared_norm(y_true, y_pred, thetas=None, mask=None):
    r"""Compute the square norm.

    There is a possibility to add a mask on which the errors are computed.
    Parameters
    ----------
    y_true:  torch.Tensor of shape (n,m)
    y_pred:  torch.Tensor of shape (n,m)
    thetas:  torch.Tensor of shape m
    mask : torch.Tensor of size (n,m), type torch.bool
    Returns
    -------
    res: torch.Tensor of shape (1)
    """
    residual = y_true - y_pred
    n, m, _ = residual.shape
    if mask is None:
        mask = torch.ones(n, m, dtype=torch.bool)
    tmp = (residual[mask]**2)
    res = tmp.sum() / n / m / 2
    return res


def ploss(y_true, y_pred, probs):
    """Compute the pinball loss.

    Parameters
    ----------
    y_true : {tensor-like}, shape = [n_samples, 1]
    y_pred : {tensor-like}, shape = [n_samples, n_quantiles, 1]
    probs : {tensor-like}, shape = [n_quantiles, 1]
    Returns
    -------
    l : {tensor}, shape = [1]
        Average loss for all quantile levels.
    """
    n, m, _ = y_pred.shape
    residual = y_true.repeat(m).reshape(m, n).T - y_pred.squeeze()
    loss = torch.max(probs.view(-1) * residual,
                     (probs - 1).view(-1) * residual).sum() / n / m
    return(loss)


def closs(y_pred):
    """Compute the crossing loss.

    Parameters
    ----------
    pred : {array-like}, shape = [n_quantiles, n_samples] or [n_samples]
        Predictions.
    Returns
    -------
    l : {array}, shape = [1]
        Average loss for all quantile leves.
    """
    n, m, _ = y_pred.shape
    res = torch.max(y_pred[:, :-1] - y_pred[:, 1:],
                    torch.zeros_like(y_pred[:, :-1]))
    return(res.mean())


def ploss_with_crossing(lbda_nc):
    """Wrap the pinball loss with crossing.

    Parameters
    ----------
    lbda_nc : float
    Returns
    -------
    ploss_with_crossing_lbda : function
    """
    def ploss_with_crossing_lbda(y_true, y_pred, probs):
        return(ploss(y_true, y_pred, probs) + lbda_nc * closs(y_pred))
    return ploss_with_crossing_lbda
