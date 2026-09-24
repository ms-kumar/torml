"""Exact Gaussian process regression with a PyTorch backend."""

from __future__ import annotations

import math

import torch

from torml.base import RegressorMixin
from torml.utils._validation import check_array, check_is_fitted, check_X_y


class GaussianProcessRegressor(RegressorMixin):
    """Exact GPR with an RBF kernel.

    Parameters
    ----------
    length_scale : float, default=1.0
        RBF kernel length scale. Must be > 0.
    alpha : float, default=1e-6
        Noise variance added to the kernel diagonal. Must be >= 0.

    Attributes
    ----------
    X_train_ : torch.Tensor of shape (n_samples, n_features)
        Training inputs.
    y_train_ : torch.Tensor of shape (n_samples,)
        Training targets.
    L_ : torch.Tensor of shape (n_samples, n_samples)
        Cholesky factor of the kernel matrix.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "GaussianProcessRegressor"

    def __init__(self, length_scale=1.0, alpha=1e-6):
        self.length_scale = length_scale
        self.alpha = alpha

    def _rbf(self, A, B):
        """RBF kernel matrix between rows of ``A`` and ``B``."""
        dist2 = torch.cdist(A, B, p=2) ** 2
        return torch.exp(-0.5 * dist2 / (float(self.length_scale) ** 2))

    def fit(self, X, y):
        """Condition the process on ``X``, ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.
        y : array-like of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : GaussianProcessRegressor
            Fitted process.
        """
        if isinstance(self.length_scale, bool) or not isinstance(
            self.length_scale, (int, float)
        ):
            raise TypeError(
                f"length_scale must be a float, got {type(self.length_scale).__name__}."
            )
        if float(self.length_scale) <= 0:
            raise ValueError(f"length_scale must be > 0, got {self.length_scale}.")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError(f"alpha must be a float, got {type(self.alpha).__name__}.")
        if float(self.alpha) < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
        X, y = check_X_y(X, y)
        X = X.to(dtype=torch.float32)
        self.X_train_ = X
        self.y_train_ = y.to(dtype=torch.float32).reshape(-1)
        self.n_features_in_ = int(X.shape[1])
        noise = float(self.alpha) + 1e-10
        base = self._rbf(X, X)
        eye = torch.eye(int(X.shape[0]))
        jitter = noise
        for _ in range(5):
            try:
                self.L_ = torch.linalg.cholesky(  # pylint: disable=not-callable
                    base + jitter * eye
                )
                break
            except RuntimeError:
                jitter = max(jitter * 100.0, 1e-8)
        else:
            raise ValueError("Kernel matrix is not positive definite.")
        self._weights = torch.cholesky_solve(
            self.y_train_.unsqueeze(1), self.L_
        ).squeeze(1)
        return self

    def predict(self, X, return_std=False):
        """Predict posterior mean (and std) at ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.
        return_std : bool, default=False
            Also return posterior standard deviations.

        Returns
        -------
        mean : torch.Tensor of shape (n_queries,)
            Posterior means.
        std : torch.Tensor of shape (n_queries,), only if ``return_std=True``
            Posterior standard deviations.
        """
        check_is_fitted(self, attributes=["L_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but the process was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        K_star = self._rbf(Xt, self.X_train_)
        mean = K_star @ self._weights
        if not return_std:
            return mean
        V = torch.cholesky_solve(K_star.T, self.L_)
        var = 1.0 - (K_star * V.T).sum(dim=1)
        return mean, torch.sqrt(var.clamp(min=0))

    def log_marginal_likelihood(self):
        """Return the log marginal likelihood of the training data.

        Returns
        -------
        lml : float
            Log marginal likelihood.
        """
        check_is_fitted(self, attributes=["L_"])
        n = int(self.X_train_.shape[0])
        data_fit = -0.5 * float(self.y_train_ @ self._weights)
        complexity = -float(torch.log(torch.diagonal(self.L_)).sum())
        const = -0.5 * n * math.log(2.0 * math.pi)
        return data_fit + complexity + const
