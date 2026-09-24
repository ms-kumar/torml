"""Gaussian random projection with a PyTorch backend."""

from __future__ import annotations

import math

import torch

from torml.base import TransformerMixin
from torml.utils._random import check_random_state
from torml.utils._validation import check_array, check_is_fitted


def johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.1) -> int:
    """Minimum dimensions for the Johnson-Lindenstrauss guarantee.

    Parameters
    ----------
    n_samples : int
        Number of samples to embed.
    eps : float, default=0.1
        Distortion tolerance in (0, 1).

    Returns
    -------
    dim : int
        Lower bound on the target dimensionality.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError(f"n_samples must be an int, got {type(n_samples).__name__}.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if not 0.0 < float(eps) < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}.")
    denom = (eps**2 / 2) - (eps**3 / 3)
    return max(1, int(math.ceil(4 * math.log(n_samples) / denom)))


class GaussianRandomProjection(TransformerMixin):
    """Project with a Gaussian random matrix (unit column scale).

    Parameters
    ----------
    n_components : int, 'auto' or None, default='auto'
        Target dimensionality. ``'auto'`` uses the Johnson-Lindenstrauss
        bound with ``eps`` (needs ``fit`` sample count).
    eps : float, default=0.1
        Distortion for ``n_components='auto'``.
    random_state : int, torch.Generator or None, default=None
        Seed for the random matrix.

    Attributes
    ----------
    components_ : torch.Tensor of shape (n_features, n_components)
        Random projection matrix.
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Target dimensionality used.
    """

    name = "GaussianRandomProjection"

    def __init__(self, n_components="auto", eps=0.1, random_state=None):
        self.n_components = n_components
        self.eps = eps
        self.random_state = random_state

    def fit(self, X, y=None):
        """Draw the random matrix for ``X``'s width.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : GaussianRandomProjection
            Fitted projector.
        """
        generator = check_random_state(self.random_state)
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        n_samples, n_features = int(Xt.shape[0]), int(Xt.shape[1])
        self.n_features_in_ = n_features
        nc = self.n_components
        if nc == "auto":
            k = johnson_lindenstrauss_min_dim(n_samples, eps=float(self.eps))
        elif nc is None:
            k = n_features
        elif isinstance(nc, bool) or not isinstance(nc, int):
            raise TypeError(
                f"n_components must be int, 'auto' or None, got {type(nc).__name__}."
            )
        elif nc < 1:
            raise ValueError(f"n_components must be >= 1, got {nc}.")
        else:
            k = int(nc)
        self.n_components_ = k
        self.components_ = torch.randn(
            n_features, k, generator=generator, dtype=torch.float32
        ) / math.sqrt(k)
        return self

    def transform(self, X):
        """Project ``X`` with the random matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, n_components_)
            Projected data.
        """
        check_is_fitted(self, attributes=["components_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but the projector was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        return Xt @ self.components_

    def _transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        """Approximately invert the projection via pseudo-inverse.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components_)
            Projected data.

        Returns
        -------
        X_orig : torch.Tensor of shape (n_samples, n_features)
            Approximate reconstruction.
        """
        check_is_fitted(self, attributes=["components_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        if int(Xt.shape[1]) != int(self.n_components_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} components, but the projector has "
                f"{int(self.n_components_)}."
            )
        return Xt @ torch.linalg.pinv(self.components_)  # pylint: disable=not-callable
