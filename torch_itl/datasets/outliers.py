import torch


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
