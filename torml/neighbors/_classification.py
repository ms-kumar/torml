"""Nearest-neighbor classification with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import ClassifierMixin
from torml.neighbors._base import (
    _validate_n_neighbors,
    _validate_p,
    _validate_weights,
    distance_weights,
    kneighbors,
)
from torml.utils._validation import check_array, check_is_fitted, check_X_y


class KNeighborsClassifier(ClassifierMixin):
    """Classifier based on k-nearest neighbors voting.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weighting for predictions: uniform majority vote or
        inverse-distance weighted vote.
    p : float, default=2
        Minkowski order for distances. ``inf`` gives Chebyshev distance.

    Attributes
    ----------
    classes_ : torch.Tensor of shape (n_classes,)
        Sorted unique training labels.
    X_ : torch.Tensor of shape (n_samples, n_features)
        Training data.
    y_ : torch.Tensor of shape (n_samples,)
        Training class indices into ``classes_``.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "KNeighborsClassifier"

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        """Fit the classifier by memorizing the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : KNeighborsClassifier
            Fitted classifier.
        """
        _validate_n_neighbors(self.n_neighbors)
        _validate_weights(self.weights)
        p = _validate_p(self.p)
        self._p = p
        X, y = check_X_y(X, y)
        if int(X.shape[0]) < int(self.n_neighbors):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be <= n_samples "
                f"({int(X.shape[0])})."
            )
        flat_y = y.reshape(-1)
        self.classes_, inverse = torch.unique(flat_y, sorted=True, return_inverse=True)
        self.classes_ = self.classes_.to(device=X.device)
        self.X_ = X
        self.y_ = inverse.to(dtype=torch.long, device=X.device)
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

    def predict_proba(self, X):
        """Return class probabilities for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        proba : torch.Tensor of shape (n_queries, n_classes)
            Per-class probabilities.
        """
        check_is_fitted(self, attributes=["X_", "y_"])
        dist, ind = self.kneighbors(X)
        neighbor_labels = self.y_[ind]
        n_queries = int(ind.shape[0])
        n_classes = int(self.classes_.shape[0])
        if self.weights == "uniform":
            counts = torch.zeros(
                n_queries, n_classes, dtype=dist.dtype, device=dist.device
            )
            counts.scatter_add_(
                1,
                neighbor_labels,
                torch.ones_like(neighbor_labels, dtype=dist.dtype),
            )
            return counts / float(self.n_neighbors)
        weights = distance_weights(dist)
        exact = dist == 0
        has_exact = exact.any(dim=1)
        proba = torch.zeros(n_queries, n_classes, dtype=dist.dtype, device=dist.device)
        if bool(has_exact.any()):
            for i in torch.where(has_exact)[0].tolist():
                exact_labels = neighbor_labels[i][exact[i]]
                counts = torch.bincount(exact_labels, minlength=n_classes).to(
                    dtype=dist.dtype
                )
                proba[i] = counts / counts.sum()
        rest = (~has_exact).nonzero().reshape(-1).tolist()
        if rest:
            r_idx = torch.tensor(rest, dtype=torch.long, device=dist.device)
            w = weights[r_idx]
            weighted = torch.zeros(
                len(rest), n_classes, dtype=dist.dtype, device=dist.device
            )
            weighted.scatter_add_(1, neighbor_labels[r_idx], w)
            proba[r_idx] = weighted / w.sum(dim=1, keepdim=True)
        return proba

    def predict(self, X):
        """Predict class labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_queries,)
            Predicted labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[torch.argmax(proba, dim=1)]
