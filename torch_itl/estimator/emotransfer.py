"""Implement the EmoTransfer class."""
import torch
from .vitl import VITL
from .utils import squared_norm
from scipy.linalg import solve_sylvester


class EmoTransfer(VITL):
    """Implement emotion transfer.

    The loss is the square loss, as proposed in
    'Emotion Transfer Using Vector-Valued Infinite Task Learning'.
    """

    def __init__(self, model, lbda, sampler, inp_emotion='joint',
                 inc_emotion=True):
        """Initialize the estimator's parameters."""
        super().__init__(model, squared_norm, lbda, sampler)
        self.inp_emotion = inp_emotion
        self.inc_emotion = inc_emotion

    def initialize(self, data):
        """Transform data into suitable x_train, y_train.

        The problem then corresponds to a kernelized ridge regression.
        Also set emotion anchors, and load it into the model.
        Parameters
        ----------
        data: torch.Tensor of shape (n_samples, n_emotions, n_landmarks)
           Input vector of data
        Returns
        -------
        Nothing
        """
        n, m, nf = data.shape

        if self.inp_emotion == 'joint':
            x_train = data.reshape(-1, nf)
            y_train = torch.zeros(m * n, m, nf)
            for i in range(m * n):
                y_train[i] = data[i // m]
            thetas = self.sampler.sample()
            self.model.m = m

        else:
            x_train = data[:, self.inp_emotion, :]
            if self.inc_emotion:
                y_train = data
                thetas = self.sampler.sample()
                self.model.m = m
            else:
                mask = [i != self.inp_emotion for i in range(m)]
                y_train = data[:, mask, :]
                thetas = self.sampler.sample()[mask]
                self.model.m = m - 1

        self.model.thetas = thetas
        self.model.n = x_train.shape[0]
        self.model.x_train = x_train
        self.model.y_train = y_train

    def risk(self, data, thetas=None):
        """Compute the risk associated to the data.

        Parameters
        ----------
        data: torch.Tensor of shape (n_samples, n_emotions, n_landmarks)
            Input vector of data
        Returns
        -------
        res: torch.Tensor of shape (1)
            risk of the predictor on the data
        """
        n, m, nf = data.shape
        if self.inp_emotion == 'joint':
            x = data.reshape(-1, nf)
            y = torch.zeros(m * n, m, nf)
            for i in range(m * n):
                y[i] = data[i // m]

        else:
            x = data[:, self.inp_emotion, :]
            if self.inc_emotion:
                y = data
            else:
                mask = [i != self.inp_emotion for i in range(m)]
                y = data[:, mask, :]

        if thetas is None:
            thetas = self.model.thetas

        pred = self.model.forward(x, thetas)
        res = self.cost(y, pred, thetas)
        return res

    def training_risk(self):
        """Compute the risk associated to the stored training data.

        Parameters
        ----------
        None
        Returns
        -------
        res: torch.Tensor of shape (1)
            risk of the predictor on the data
        """
        if not hasattr(self.model, 'x_train'):
            raise Exception('No training data provided')
        pred = self.model.forward(self.model.x_train, self.model.thetas)
        res = self.cost(self.model.y_train, pred, self.model.thetas)
        return res

    def fit(self, data, verbose=False):
        """Fit the emotion transfer model by a closed form solution.

        The matrix A of the model has to be invertible
        Parameters
        ----------
        data: torch.Tensor of shape (n_samples, n_emotions, n_landmarks)
            Input vector of data
        verbose: Bool
            some prints along the way (or not)
        Returns
        -------
        Nothing
        """
        if verbose:
            print('Initialize data')
        self.initialize(data)
        self.model.compute_gram_train()

        if verbose:
            print('Solving the linear system')

        if torch.norm(self.model.A - torch.eye(self.model.output_dim)) < 1e-10:
            tmp = self.model.G_xt + self.lbda * \
                self.model.n * self.model.m * \
                torch.eye(self.model.n * self.model.m)
            alpha, _ = torch.solve(
                self.model.y_train.reshape(-1, self.model.output_dim), tmp)
            self.model.alpha = alpha.reshape(self.model.n, self.model.m, -1)

        else:
            B = torch.inverse(self.model.A).numpy()
            Q = (self.model.y_train.reshape(
                -1, self.model.output_dim)).numpy() @ B
            alpha_np = solve_sylvester(
                self.model.G_xt, self.lbda * self.model.n * self.model.m * B,
                Q)
            self.model.alpha = torch.from_numpy(
                alpha_np).reshape(self.model.n, self.model.m, -1)

        if verbose:
            print('Coefficients alpha fitted, empirical risk=',
                  self.training_risk())

    def fit_partial(self, data, mask, verbose=False):
        """Fit the emotion transfer model on partial data.

        Missing data are encoded in a mask, solver is closed-form.
        Parameters
        ----------
        data: torch.Tensor of shape (n_samples, n_emotions, n_landmarks)
           Input vector of data
        mask: torch.Tensor(dtype=torch.bool) of shape (n_samples, n_emotions)
        Returns
        -------
        Nothing
        """
        if verbose:
            print('Initialize data & compute gram matrices')

        if self.inp_emotion == 'joint':
            x_train = data[mask]
            n, nf = x_train.shape
            m = data.shape[1]
            self.model.m = m
            thetas = self.sampler.sample()
            y_train = torch.zeros(n, m, nf)
            output_mask = torch.zeros((n, m), dtype=torch.bool)
            count = 0
            for i in range(data.shape[0]):
                for j in range(m):
                    if mask[i, j].item():
                        y_train[count] = data[i]
                        output_mask[count] = mask[i]
                        count += 1

        else:
            x_train = data[:, self.inp_emotion, :][mask[:, self.inp_emotion]]
            n, nf = x_train.shape
            m = data.shape[1]
            if self.inc_emotion:
                self.model.m = m
                thetas = self.sampler.sample()
                y_train = data[mask[:, self.inp_emotion]]
                output_mask = torch.zeros((n, m), dtype=torch.bool)
                count = 0
                for i in range(data.shape[0]):
                    for j in range(m):
                        if mask[i, self.inp_emotion] and mask[i, j]:
                            output_mask[count, j] = True
                    count += 1
            else:
                mask_m = [i != self.inp_emotion for i in range(m)]
                y_train = data[mask[:, self.inp_emotion]][:, mask_m, :]
                output_mask = torch.zeros((n, m-1), dtype=torch.bool)
                count = 0
                for i in range(data.shape[0]):
                    for j in range(m):
                        if (mask[i, self.inp_emotion] and mask[i, j] and
                                j != self.inp_emotion):
                            output_mask[count, j] = True
                    count += 1
                output_mask = output_mask[:, mask_m]
                self.model.m = m - 1
                thetas = self.sampler.sample()[mask_m]

        self.model.x_train = x_train
        self.model.y_train = y_train
        self.model.n = x_train.shape[0]
        self.model.thetas = thetas
        self.model.output_mask = output_mask

        self.model.compute_gram_train()

        if verbose:
            print("Solving the linear system")

        tmp = self.model.G_xt + self.lbda * self.model.n * \
            self.model.m * torch.eye(self.model.n * self.model.m)
        output_mask_ravel = output_mask.reshape(-1)
        tmp = tmp[output_mask_ravel][:, output_mask_ravel]

        alpha_sol, _ = torch.solve(self.model.y_train[output_mask], tmp)

        self.model.alpha = torch.zeros(self.model.n, self.model.m, nf)

        self.model.alpha[output_mask] = alpha_sol

    def fit_dim_red(self, data, r, eigen=None, verbose=False):
        """Fit the emotion transfer model with dimensionality reduction.

        The solver is closed-form, and benefits from a low rank replacement
        of the matrix A based on SVD of the data covariance.
        If Y^T Y = V S V^T then uses A = V diag( eigen, 0) V^T
        Parameters
        ----------
        data: torch.Tensor of shape (n_samples, n_emotions, n_landmarks)
           Input vector of data
        r: Int
            Rank of A
        eigen: torch.Tensor of shape (r)
            Eigenvalues of A to consider. Default None -> use all eigen=1
        Returns
        -------
        Nothing
        """
        if verbose:
            print('Computing Gram matrices')
        self.initialize(data)
        self.model.compute_gram_train()

        tmp = self.model.G_xt + self.lbda * self.model.n * self.model.m * \
            torch.eye(self.model.n * self.model.m)

        if verbose:
            print('Computing SVD of empirical covariance')
        cor = 1 / self.model.n * \
            self.model.y_train.reshape(
                -1, self.model.output_dim).T @ self.model.y_train.reshape(
                    -1, self.model.output_dim)
        u, d, v = torch.svd(cor)

        if verbose:
            print("Solving the associated linear system")

        identity_r = torch.diag_embed(
            torch.Tensor([1 for i in range(r)] + [0 for i in range(
                self.model.output_dim - r)]))
        proj_r = identity_r[:, :r]

        if eigen is None:

            gamma_r, _ = torch.solve(
                self.model.y_train.reshape(
                    -1, self.model.output_dim) @ v @ proj_r, tmp)
            self.model.alpha = gamma_r @ proj_r.T @ v.T
            self.model.A = v @ identity_r @ v.T

        else:
            B = torch.inverse(torch.diag(eigen)).numpy()
            Q = (self.model.y_train.reshape(-1, self.model.output_dim)
                 @ v @ proj_r).numpy() @ B
            gamma_r = solve_sylvester(
                self.model.G_xt, self.lbda * self.model.n * self.model.m * B,
                Q)
            self.model.alpha = torch.from_numpy(
                gamma_r) @ torch.diag(eigen) @ proj_r.T @ v.T
            self.model.A = v @ proj_r @ torch.diag(eigen) @ proj_r.T @ v.T

        if verbose:
            print('Coefficients alpha fitted, empirical risk=',
                  self.training_risk())
