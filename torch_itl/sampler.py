import torch


class LinearSampler(object):
    'Linear sampling class'

    def __init__(self, a, b, m=None, epsilon=None):
        self.a = a
        self.b = b
        if epsilon is None:
            self.epsilon = 0.01
        else:
            self.epsilon = epsilon

    def sample(self, m=None):
        if m is None:
            if not hasattr(self, 'm'):
                raise ValueError('No value given for m')
            else:
                return torch.linspace(self.a + self.epsilon, self.b - self.epsilon, self.m).view(-1, 1)
        else:
            return torch.linspace(self.a + self.epsilon, self.b - self.epsilon, m).view(-1, 1)


# TODO: Currently hand coded values ar returned, modify class later
class CircularSampler(object):
    def __init__(self, a=0, b=torch.acos(torch.zeros(1)).item()*4, m=None):
        self.a = a
        self.b = b
        self.m = m

    def sample(self, m):
        # angles = torch.linspace(a, b, m)
        # return torch.stack((torch.cos(angles), torch.sin(angles))).T
        # Just for now, to avoid errors and keep the order in check
        assert (m == self.m)
        return torch.tensor([[1/2, -(3**0.5)/2], [1, 0], [-1, 0]], dtype=torch.float)
