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
                return torch.linspace(
                    self.a + self.epsilon, self.b - self.epsilon,
                    self.m).view(-1, 1)
        else:
            return torch.linspace(
                self.a + self.epsilon, self.b - self.epsilon, m).view(-1, 1)
