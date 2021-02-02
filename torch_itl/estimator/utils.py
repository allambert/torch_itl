import torch


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
    return torch.where(alpha.abs() - tau < 0, torch.zeros_like(alpha), alpha.abs() - tau) * torch.sign(alpha)


def squared_norm(y_true, y_pred, thetas, mask=None):
    residual = y_true - y_pred
    n, m, _ = residual.shape
    if mask is None:
        mask = torch.ones(n, m, dtype=torch.bool)
    tmp = (residual[mask]**2)
    return tmp.sum() / n / m / 2
