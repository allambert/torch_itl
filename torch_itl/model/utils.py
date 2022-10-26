"""Utilitary funnctions for the model classes."""
import torch


def kron(matrix1, matrix2):
    """Compute the kronecker product of gram matrices.

    Parameters
    ----------
    matrix1:  torch.Tensor of shape (n_1, n_1)
    matrix2:  torch.Tensor of shape (n_2, n_2)
    Returns
    -------
    output: torch.Tensor of shape (n_1*n_2, n_1*n_2)
        Kronecker product of matrix1 and matrix2.
    """
    output = torch.ger(
        matrix1.view(-1),
        matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
        [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0),
                              matrix1.size(1) * matrix2.size(1))
    return output
