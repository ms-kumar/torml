"""Empirical covariance estimation with a PyTorch backend."""

from __future__ import annotations

import math

import torch

from torml.base import BaseEstimator
from torml.utils._validation import check_array, check_is_fitted


class EmpiricalCovariance(BaseEstimator):
    """Maximum-likelihood covariance estimator.

    Attributes
    ----------
    location_ : torch.Tensor of shape (n_features,)
        Empirical mean.
    covariance_ : torch.Tensor of shape (n_features, n_features)
        Empirical covariance (biased, divided by n_samples).
    precision_ : torch.Tensor of shape (n_features, n_features)
        Pseudo-inverse of the covariance.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "EmpiricalCovariance"

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Estimate mean and covariance of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : EmpiricalCovariance
            Fitted estimator.
        """
        Xt = check_array(X, ensure_2d=True)
        self.location_ = Xt.mean(dim=0)
        centered = Xt - self.location_
        self.covariance_ = (centered.T @ centered) / int(Xt.shape[0])
        self.precision_ = torch.linalg.pinv(  # pylint: disable=not-callable
            self.covariance_
        )
        self.n_features_in_ = int(Xt.shape[1])
        return self

    def mahalanobis(self, X):
        """Return squared Mahalanobis distances for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        dist : torch.Tensor of shape (n_samples,)
            Squared distances.
        """
        check_is_fitted(self, attributes=["precision_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but the estimator was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        diff = Xt - self.location_
        return torch.sum((diff @ self.precision_) * diff, dim=1)

    def score(self, X, y=None):
        """Return mean Gaussian log likelihood of ``X``.

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
        check_is_fitted(self, attributes=["precision_"])
        Xt = check_array(X, ensure_2d=True)
        d = int(Xt.shape[1])
        sign, log_det = torch.slogdet(
            self.covariance_
            + 1e-12
            * torch.eye(d, dtype=self.covariance_.dtype, device=self.covariance_.device)
        )
        if int(sign.item()) <= 0:
            log_det = torch.tensor(
                float("-inf"), dtype=log_det.dtype, device=log_det.device
            )
        ll = -0.5 * (d * math.log(2.0 * math.pi) + log_det + self.mahalanobis(Xt))
        return float(ll.mean())
