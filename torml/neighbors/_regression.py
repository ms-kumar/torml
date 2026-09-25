"""Nearest-neighbor regression with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import RegressorMixin
from torml.neighbors._base import (
    _validate_n_neighbors,
    _validate_p,
    _validate_weights,
    distance_weights,
    kneighbors,
)
from torml.utils._validation import check_array, check_is_fitted, check_X_y


class KNeighborsRegressor(RegressorMixin):
    """Regressor based on k-nearest neighbors averaging.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weighting for predictions: uniform mean or inverse-distance
        weighted mean.
    p : float, default=2
        Minkowski order for distances. ``inf`` gives Chebyshev distance.

    Attributes
    ----------
    X_ : torch.Tensor of shape (n_samples, n_features)
        Training data.
    y_ : torch.Tensor of shape (n_samples,)
        Training targets.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "KNeighborsRegressor"

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        """Fit the regressor by memorizing the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNeighborsRegressor
            Fitted regressor.
        """
        _validate_n_neighbors(self.n_neighbors)
        _validate_weights(self.weights)
        p = _validate_p(self.p)
        self._p = p
        X, y = check_X_y(X, y)
        y = y.to(dtype=X.dtype).reshape(-1)
        if int(X.shape[0]) < int(self.n_neighbors):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be <= n_samples "
                f"({int(X.shape[0])})."
            )
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = int(X.shape[1])
        self.n_samples_in_ = int(X.shape[0])
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find neighbors of ``X`` in the training set.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.
        n_neighbors : int or None, default=None
            Overrides the fitted ``n_neighbors`` for this call.
        return_distance : bool, default=True
            If False, return indices only.

        Returns
        -------
        dist : torch.Tensor, only if ``return_distance=True``
            Distances, shape (n_queries, n_neighbors).
        ind : torch.Tensor of shape (n_queries, n_neighbors)
            Indices into the training set.
        """
        check_is_fitted(self, attributes=["X_", "y_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but "
                f"{type(self).__name__} was fitted with "
                f"{int(self.n_features_in_)} features."
            )
        k = (
            self.n_neighbors
            if n_neighbors is None
            else _validate_n_neighbors(n_neighbors)
        )
        dist, ind = kneighbors(self.X_, Xt, k, p=self._p)
        if return_distance:
            return dist, ind
        return ind

    def predict(self, X):
        """Predict targets for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_queries,)
            Predicted values.
        """
        check_is_fitted(self, attributes=["X_", "y_"])
        dist, ind = self.kneighbors(X)
        neighbor_vals = self.y_[ind]
        if self.weights == "uniform":
            return neighbor_vals.mean(dim=1)
        exact = dist == 0
        has_exact = exact.any(dim=1)
        out = torch.empty(
            int(ind.shape[0]), dtype=neighbor_vals.dtype, device=neighbor_vals.device
        )
        if bool(has_exact.any()):
            for i in torch.where(has_exact)[0].tolist():
                out[i] = neighbor_vals[i][exact[i]].mean()
        rest = (~has_exact).nonzero().reshape(-1).tolist()
        if rest:
            r_idx = torch.tensor(rest, dtype=torch.long, device=neighbor_vals.device)
            w = distance_weights(dist[r_idx])
            out[r_idx] = (neighbor_vals[r_idx] * w).sum(dim=1) / w.sum(dim=1)
        return out
