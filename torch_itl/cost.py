import torch


def ploss(y_true, y_pred, probs):
    """Computes the pinball loss.
    Parameters
    ----------
    y_true : {tensor-like}, shape = [n_samples]
    y_pred : {tensor-like}, shape = [n_samples,n_quantiles]
    probs : {tensor-like}, shape = [n_quantiles]
    Returns
    -------
    l : {tensor}, shape = [1]
        Average loss for all quantile levels.
    """
    residual = y_true.view(-1, 1) - y_pred
    n, m = residual.shape
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
    n, m = y_pred.shape
    res = torch.max(y_pred[:, :-1] - y_pred[:, 1:],
                    torch.zeros_like(y_pred[:, :-1]))
    return(res.sum())


def ploss_with_crossing(lbda_nc):

    def ploss_with_crossing_lbda(y_true, y_pred, probs):
        return(ploss(y_true, y_pred, probs) + lbda_nc * closs(y_pred))
    return ploss_with_crossing_lbda
