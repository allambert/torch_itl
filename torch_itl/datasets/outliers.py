import torch
import numpy as np

from .synthetic_func_or import SyntheticGPmixture



def add_type1_outliers(X, freq_sample=0.02, seed=443, coef=-1., alternate_coef=False):
    n, m = X.shape
    res = X.detach().clone()
    n_contaminated = int(freq_sample * n)
    np.random.seed(seed)
    contaminated_inds = torch.from_numpy(np.random.choice(np.arange(n), n_contaminated, replace=False))
    ind_shift = contaminated_inds.flip(0)
    if alternate_coef :
        coef = torch.tensor([(-1) ** i * coef for i in range(n_contaminated)]).unsqueeze(1)
    res[contaminated_inds] = coef * X[ind_shift]
    return res


def add_type2_outliers(X, freq_sample=0.02, seed=443, seed_gps=56, 
                       covs_params=(0.01, 0.05, 1, 4), scale=2, intensity=2.5, additive=True):
    n, m = X.shape
    res = X.detach().clone()
    theta = torch.linspace(0, 1, m)
    gp_outliers =  SyntheticGPmixture(len(covs_params), (covs_params, covs_params), noise=(None, None), scale=scale)
    gp_outliers.drawGP(theta, seed_gp=seed_gps)
    n_contaminated = int(freq_sample * n)
    _, drawns_gps = gp_outliers.sample(n_contaminated, new_GP=False, seed_gp=seed_gps, seed_coefs=seed)
    np.random.seed(seed)
    contaminated_inds = torch.from_numpy(np.random.choice(np.arange(n), n_contaminated, replace=False))
    if additive:
        res[contaminated_inds] += intensity * drawns_gps
    else:
        res[contaminated_inds] = intensity * drawns_gps
    return res


def add_type3_outliers(X, freq_sample=1., freq_loc=0.1, intensity=0.5, seed=453):
    n, m = X.shape
    res = X.detach().clone()
    a_max = (X.abs().max()) * intensity
    n_contaminated = int(freq_sample * n)
    np.random.seed(seed)
    contaminated_inds = np.random.choice(np.arange(n), n_contaminated, replace=False)
    for i in contaminated_inds:
        m_contaminated = int(freq_loc * m)
        contaminated_locs = np.random.choice(np.arange(m), m_contaminated, replace=False)
        noise = np.random.uniform(- a_max.item(), a_max.item(), m_contaminated)
        res[i, contaminated_locs] = torch.from_numpy(noise)
    return res


def add_local_outliers(X, freq_sample=0.5, freq_loc=0.2, std=0.3):
    n, d = X.shape
    res = X.clone()
    mask_samples = (torch.rand(n) < freq_sample)
    for i in range(n):
        if mask_samples[i]:
            mask_loc = (torch.rand(d) < freq_loc)
            noise = std * torch.randn(d)
            res[i] += mask_loc * noise
    return res


def add_global_outliers_worse(X, Y, n_o, intensity=1, seed='fixed'):
    n, d = X.shape
    x_o = torch.zeros(n_o, d)
    y_o = torch.zeros(n_o, d)
    if seed == 'fixed':
        id_o = [7, 9, 18, 26]
        x_o = X[id_o]
        y_o = - intensity * Y[id_o]
    else:
        perm = torch.randperm(n, generator=torch.manual_seed(seed))
        for i in range(n_o):
            x_o[i] = X[perm[i]]
            y_o[i] = - intensity * Y[perm[i]]
    return torch.cat((X, x_o), 0), torch.cat((Y, y_o), 0)


def add_global_outliers_linear(X, Y, n_o, steep_x=None, steep_y=None):
    n, d = X.shape
    t = torch.linspace(0, 1, d).expand(n_o, d)
    if steep_x is None:
        steep_x = 2*torch.randn(n_o)
    if steep_y is None:
        steep_y = 2*torch.randn(n_o)
    x_o = (steep_x * t.T).T
    y_o = (steep_y * t.T).T
    return torch.cat((X, x_o), 0), torch.cat((Y, y_o), 0)


# # %%
# n = 100
# d = 100
# X = torch.randn(n, d)
# Y = torch.randn(n, d)
# freq_sample = 0.2
# freq_loc = 0.1
# std = 2
# X_new = local_outliers(X, freq_sample, freq_loc, std)
# X_aug, Y_aug = global_outliers_linear(X, Y, 4)
#
# X_aug.shape
