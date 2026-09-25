"""Principal Component Analysis with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import TransformerMixin
from torml.utils._validation import check_array, check_is_fitted


class PCA(TransformerMixin):
    """PCA via full SVD (``torch.linalg.svd``).

    Parameters
    ----------
    n_components : int, float or None, default=None
        Components to keep: int counts, float in (0, 1) selects by variance
        ratio, None keeps all (min(n_samples, n_features)).

    Attributes
    ----------
    components_ : torch.Tensor of shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance_ : torch.Tensor of shape (n_components,)
        Variance per component.
    explained_variance_ratio_ : torch.Tensor of shape (n_components,)
        Fraction of total variance per component.
    singular_values_ : torch.Tensor of shape (n_components,)
        Singular values.
    mean_ : torch.Tensor of shape (n_features,)
        Feature means.
    n_features_in_ : int
        Number of features seen during fit.
    n_samples_seen_ : int
        Number of samples seen during fit.
    """

    name = "PCA"

    def __init__(self, n_components=None):
        self.n_components = n_components

    def _resolve_n_components(self, n_samples: int, n_features: int) -> int:
        """Validate ``n_components`` against data dimensions."""
        max_comp = min(n_samples, n_features)
        nc = self.n_components
        if nc is None:
            return max_comp
        if isinstance(nc, bool):
            raise TypeError("n_components must be int, float or None, got bool.")
        if isinstance(nc, int):
            if not 1 <= nc <= max_comp:
                raise ValueError(
                    f"n_components={nc} must satisfy 1 <= n_components <= "
                    f"min(n_samples, n_features)={max_comp}."
                )
            return nc
        if isinstance(nc, float):
            if not 0.0 < nc < 1.0:
                raise ValueError(f"n_components as float must be in (0, 1), got {nc}.")
            return max_comp
        raise TypeError(
            "n_components must be int, float or None, " f"got {type(nc).__name__}."
        )

    def fit(self, X, y=None):
        """Compute principal components of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : PCA
            Fitted transformer.
        """
        Xt = check_array(X, ensure_2d=True)
        n_samples, n_features = int(Xt.shape[0]), int(Xt.shape[1])
        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples, got {n_samples}.")
        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples
        self.mean_ = Xt.mean(dim=0)
        centered = Xt - self.mean_
        _, singular, Vt = torch.linalg.svd(  # pylint: disable=not-callable
            centered, full_matrices=False
        )
        total_var = float((singular * singular).sum() / (n_samples - 1))
        k = self._resolve_n_components(n_samples, n_features)
        if isinstance(self.n_components, float):
            ratios = (singular * singular) / (singular * singular).sum()
            k = (
                int(
                    torch.searchsorted(
                        torch.cumsum(ratios, dim=0), float(self.n_components)
                    ).item()
                )
                + 1
            )
            k = max(1, min(k, len(singular)))
        self.components_ = Vt[:k]
        self.singular_values_ = singular[:k]
        self.explained_variance_ = (singular[:k] ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var
            if total_var > 0
            else torch.zeros_like(self.explained_variance_)
        )
        self.n_components_ = k
        return self

    def transform(self, X):
        """Project ``X`` onto the principal axes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, n_components_)
            Projected data.
        """
        check_is_fitted(self, attributes=["components_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but PCA was fitted "
                f"with {int(self.n_features_in_)} features."
            )
        return (Xt - self.mean_) @ self.components_.T

    def _transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        """Map projected data back to feature space.

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
        Xt = check_array(X, ensure_2d=True, ensure_min_features=1)
        if int(Xt.shape[1]) != int(self.components_.shape[0]):
            raise ValueError(
                f"X has {int(Xt.shape[1])} components, but PCA has "
                f"{int(self.components_.shape[0])}."
            )
        return Xt @ self.components_ + self.mean_
