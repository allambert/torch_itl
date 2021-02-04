import sys
import torch
import sys

def proj(alpha, kappa):
    norm = torch.sqrt(torch.sum(alpha**2, axis=1))
    mask = torch.where(norm > kappa)
    alpha[mask] *= kappa / norm[mask].reshape((-1, 1))
    return alpha


def bst(alpha, tau):
    norm = (alpha**2).sum(1).sqrt()
    mask_st = torch.where(norm >= tau)
    mask_ze = torch.where(norm < tau)
    alpha[mask_st] = alpha[mask_st] - alpha[mask_st] / \
        norm[mask_st].reshape((-1, 1)) * tau
    alpha[mask_ze] = 0
    return(alpha)


def iht(alpha, tau):
    return torch.where(alpha.abs() - tau < 0,
                       torch.zeros_like(alpha),
                       alpha.abs() - tau) * torch.sign(alpha)


def squared_norm(y_true, y_pred, thetas, mask=None):
    residual = y_true - y_pred
    n, m, _ = residual.shape
    if mask is None:
        mask = torch.ones(n, m, dtype=torch.bool)
    tmp = (residual[mask]**2)
    return tmp.sum() / n / m / 2


def ploss(y_true, y_pred, probs):
    """Computes the pinball loss.
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


def ploss_dual(res, thetas):
    condition = (res < thetas) & (res > thetas - 1)
    return torch.where(condition, 0, sys.float_info.max)


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
    return(res.sum())


def ploss_with_crossing(lbda_nc):

    def ploss_with_crossing_lbda(y_true, y_pred, probs):
        return(ploss(y_true, y_pred, probs) + lbda_nc * closs(y_pred))
    return ploss_with_crossing_lbda
