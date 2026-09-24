"""Gaussian mixture models with a PyTorch backend (EM, full covariances)."""

from __future__ import annotations

import math

import torch

from torml.base import ClusterMixin
from torml.utils._random import check_random_state
from torml.utils._validation import check_array, check_is_fitted


class GaussianMixture(ClusterMixin):
    """Mixture of Gaussians fitted with expectation-maximization.

    Parameters
    ----------
    n_components : int, default=1
        Number of mixture components.
    max_iter : int, default=100
        Maximum EM iterations.
    tol : float, default=1e-3
        Convergence tolerance on mean log likelihood gain.
    random_state : int, torch.Generator or None, default=None
        Seed for choosing initial responsibilities.

    Attributes
    ----------
    weights_ : torch.Tensor of shape (n_components,)
        Mixture weights.
    means_ : torch.Tensor of shape (n_components, n_features)
        Component means.
    covariances_ : torch.Tensor of shape (n_components, n_features, n_features)
        Component covariances.
    converged_ : bool
        Whether EM converged before ``max_iter``.
    n_iter_ : int
        Iterations run.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "GaussianMixture"

    def __init__(self, n_components=1, max_iter=100, tol=1e-3, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _estimate_log_gaussian(self, X):
        """Log density of each sample under each component."""
        n, d = int(X.shape[0]), int(X.shape[1])
        out = torch.empty(n, self._k, dtype=torch.float32)
        for c in range(self._k):
            diff = X - self.means_[c]
            try:
                L = torch.linalg.cholesky(  # pylint: disable=not-callable
                    self.covariances_[c]
                )
            except RuntimeError as e:
                raise ValueError(f"Covariance of component {c} is singular.") from e
            log_det = 2.0 * float(torch.log(torch.diagonal(L)).sum())
            sol = torch.cholesky_solve(diff.T, L).T
            out[:, c] = -0.5 * (
                d * math.log(2.0 * math.pi) + log_det + (sol * diff).sum(dim=1)
            )
        return out

    def fit(self, X, y=None):
        """Run EM from a random responsibility initialization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : GaussianMixture
            Fitted mixture.
        """
        if isinstance(self.n_components, bool) or not isinstance(
            self.n_components, int
        ):
            raise TypeError(
                f"n_components must be an int, got {type(self.n_components).__name__}."
            )
        if int(self.n_components) < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise TypeError(
                f"max_iter must be an int, got {type(self.max_iter).__name__}."
            )
        if int(self.max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        generator = check_random_state(self.random_state)
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        n, d = int(Xt.shape[0]), int(Xt.shape[1])
        if n < int(self.n_components):
            raise ValueError(
                f"n_samples ({n}) must be >= n_components ({self.n_components})."
            )
        self._k = int(self.n_components)
        self.n_features_in_ = d

        resp = torch.rand(n, self._k, generator=generator) + 1e-6
        resp = resp / resp.sum(dim=1, keepdim=True)
        prev_ll: float | None = None
        self.converged_ = False
        for it in range(1, int(self.max_iter) + 1):
            nk = resp.sum(dim=0) + 1e-10
            self.weights_ = nk / n
            self.means_ = (resp.T @ Xt) / nk.unsqueeze(1)
            covs = torch.empty(self._k, d, d, dtype=torch.float32)
            for c in range(self._k):
                diff = Xt - self.means_[c]
                covs[c] = (diff * resp[:, c].unsqueeze(1)).T @ diff / nk[c]
                covs[c] = covs[c] + 1e-6 * torch.eye(d)
            self.covariances_ = covs
            log_resp = self._estimate_log_gaussian(Xt) + torch.log(
                self.weights_
            ).unsqueeze(0)
            ll = float(torch.logsumexp(log_resp, dim=1).mean())
            self.n_iter_ = it
            if prev_ll is not None and abs(ll - prev_ll) < float(self.tol):
                self.converged_ = True
                break
            prev_ll = ll
            resp = torch.softmax(log_resp, dim=1)
        return self

    def predict_proba(self, X):
        """Return component responsibilities for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        resp : torch.Tensor of shape (n_samples, n_components)
            Responsibilities.
        """
        check_is_fitted(self, attributes=["weights_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but the mixture was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        log_resp = self._estimate_log_gaussian(Xt) + torch.log(self.weights_).unsqueeze(
            0
        )
        return torch.softmax(log_resp, dim=1)

    def predict(self, X):
        """Assign each sample to the likeliest component.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
            Component indices.
        """
        return torch.argmax(self.predict_proba(X), dim=1)

    def fit_predict(self, X, y=None):
        """Fit and return training component assignments.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
            Component indices.
        """
        return self.fit(X, y).predict(X)

    def score(self, X, y=None):
        """Return mean log likelihood of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        ll : float
            Mean log likelihood.
        """
        check_is_fitted(self, attributes=["weights_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        log_resp = self._estimate_log_gaussian(Xt) + torch.log(self.weights_).unsqueeze(
            0
        )
        return float(torch.logsumexp(log_resp, dim=1).mean())
