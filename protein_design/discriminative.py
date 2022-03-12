from typing import Tuple, NamedTuple
import itertools
import torch.nn as nn
import numpy as np

from protein_design.constants import BLOSUM


class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()

        self.data_dim = kwargs["data_dim"]
        self.hid_dim = kwargs["hid_dim"]
        self.loss_fn = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Linear(self.data_dim, self.hid_dim),
            nn.ELU(),
            nn.Linear(self.hid_dim, 1),
            nn.Tanh(),
        )

    def forward(self, X):
        return self.model(X).squeeze()

    def loss(self, X, y):
        y_pred = self.model(X)
        return self.loss_fn(y, y_pred)


class GPParams(NamedTuple):
    homo_noise: float
    beta: float
    c: float
    d: float


class SequenceGP:
    def __init__(self, homo_noise=0.1, beta=0.1, c=1, d=2):
        self.params = GPParams(homo_noise, beta, c, d)
        self.K = None

    def kernel(self, Xi, Xj) -> float:
        """Return similarity of two amino acid sequences

        Parameters
        ----------
        Xi
            Integer encoded amino acid sequence
        Xj
            Integer encoded amino acid sequence

        Returns
        -------
            K(Xi, Xj) (measure of similarity between Xi and Xj)
        """
        beta = self.params.beta
        c = self.params.c
        d = self.params.d
        kij = np.prod(BLOSUM[(Xi, Xj)] ** beta)
        kii = np.prod(BLOSUM[(Xi, Xi)] ** beta)
        kjj = np.prod(BLOSUM[(Xj, Xj)] ** beta)
        k = kij / (np.sqrt(kii * kjj))
        k = np.exp(c * k)
        k = (k + c) / d
        return k

    def _fill_K(self):
        self.K = np.zeros((self.N, self.N))
        homo_noise = self.params.homo_noise
        for i in range(self.N):
            for j in range(i, self.N):
                kij = self.kernel(self.X[i], self.X[j])
                if i == j:
                    kij += homo_noise
                self.K[i, j] = kij
                self.K[j, i] = kij

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Gaussian Process model to data

        Parameters
        ----------
        X
            Array representing amino acid sequences. Each entry is an integer-encoded sequence
        y
            Scalar values to predict, one per sequence
        """
        self.X = X
        self.y = y.reshape(-1, 1)
        self.N = len(X)
        self._fill_K()

    def predict(self, Xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict scalar property values for sequences Xstar

        Parameters
        ----------
        Xstar
            Array representing amino acid sequences. Each entry is an integer-encoded sequence

        Returns
        -------
            Tuple: mean and variance of prediction
        """
        M = len(Xstar)
        Kstar = np.zeros((M, self.N))
        for i, j in itertools.product(range(M), range(self.N)):
            kij = self.kernel(Xstar[i], self.X[j])
            Kstar[i, j] = kij
        Kstarstar = np.zeros((M, M))
        for i, j in itertools.product(range(M), range(M)):
            kij = self.kernel(Xstar[i], Xstar[j])
            Kstarstar[i, j] = kij
        Kinv = np.linalg.inv(self.K)
        mu_star = np.matmul(Kstar, np.matmul(Kinv, self.y))
        sigma_star = Kstarstar - np.linalg.multi_dot([Kstar, Kinv, Kstar.T])
        return mu_star.squeeze(), np.sqrt(np.diag(sigma_star)).squeeze()
