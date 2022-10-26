"""Implement various samplers."""
from abc import ABC, abstractmethod
import torch


class Sampler(ABC):
    """Base class for samplers."""

    @abstractmethod
    def sample(self, n_samples):
        """Abstract sample."""
        pass


# class UniformSampler(Sampler):
#     'Uniform sampling class'
#
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#     def sample(self, m=None):
#         if m is None:
#             if not hasattr(self, 'm'):
#                 raise ValueError('No value given for m')
#             else:
#                 return 0.5*(self.a + self.b) + torch.rand(
#                     self.a + self.epsilon, self.b - self.epsilon,
#                     self.m).view(-1, 1)
#         else:
#             return torch.linspace(
#                 self.a + self.epsilon, self.b - self.epsilon, m).view(-1, 1)


class LinearSampler(Sampler):
    """Linear sampling class"""

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
