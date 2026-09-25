"""K-Means clustering with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import ClusterMixin
from torml.utils._random import check_random_state
from torml.utils._validation import check_array, check_is_fitted


class KMeans(ClusterMixin):
    """K-Means clustering.

    Lloyd iterations from random data-point initialization, best of
    ``n_init`` restarts by inertia.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    max_iter : int, default=300
        Maximum Lloyd iterations per restart.
    tol : float, default=1e-4
        Convergence tolerance on center shift (Frobenius norm).
    random_state : int, torch.Generator or None, default=None
        Seed for center initialization.
    n_init : int, default=10
        Number of restarts; the lowest-inertia result is kept.

    Attributes
    ----------
    cluster_centers_ : torch.Tensor of shape (n_clusters, n_features)
        Final centers.
    labels_ : torch.Tensor of shape (n_samples,)
        Index of the closest center per sample.
    inertia_ : float
        Sum of squared distances to the assigned center.
    n_iter_ : int
        Iterations run in the best restart.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "KMeans"

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state=None,
        n_init: int = 10,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

    def _validate_hyperparams(self) -> None:
        if isinstance(self.n_clusters, bool) or not isinstance(self.n_clusters, int):
            raise TypeError(
                f"n_clusters must be an int, got {type(self.n_clusters).__name__}."
            )
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise TypeError(
                f"max_iter must be an int, got {type(self.max_iter).__name__}."
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if isinstance(self.tol, bool) or not isinstance(self.tol, (int, float)):
            raise TypeError(f"tol must be a float, got {type(self.tol).__name__}.")
        if float(self.tol) < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}.")
        if isinstance(self.n_init, bool) or not isinstance(self.n_init, int):
            raise TypeError(f"n_init must be an int, got {type(self.n_init).__name__}.")
        if self.n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {self.n_init}.")

    def fit(self, X, y=None):
        """Compute cluster centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : KMeans
            Fitted clusterer.
        """
        self._validate_hyperparams()
        generator = check_random_state(self.random_state)
        Xt = check_array(X, ensure_2d=True)
        n_samples = int(Xt.shape[0])
        if n_samples < int(self.n_clusters):
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})."
            )
        self.n_features_in_ = int(Xt.shape[1])

        best_inertia: float | None = None
        for _ in range(int(self.n_init)):
            perm = torch.randperm(n_samples, generator=generator, device=Xt.device)
            centers = Xt[perm[: int(self.n_clusters)]].clone()
            assign = torch.zeros(n_samples, dtype=torch.long, device=Xt.device)
            n_iter = 0
            for n_iter in range(1, int(self.max_iter) + 1):
                dist = torch.cdist(Xt, centers, p=2)
                assign = torch.argmin(dist, dim=1)
                new_centers = centers.clone()
                for k in range(int(self.n_clusters)):
                    members = Xt[assign == k]
                    if int(members.shape[0]) > 0:
                        new_centers[k] = members.mean(dim=0)
                shift = float(torch.sum((new_centers - centers) ** 2).sqrt())
                centers = new_centers
                if shift <= float(self.tol):
                    break
            dist = torch.cdist(Xt, centers, p=2)
            assign = torch.argmin(dist, dim=1)
            inertia = float(torch.sum((Xt - centers[assign]) ** 2))
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers.clone()
                self.labels_ = assign.clone()
                self.inertia_ = inertia
                self.n_iter_ = n_iter
        return self

    def predict(self, X):
        """Assign each sample to the closest center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
            Closest-center indices.
        """
        check_is_fitted(self, attributes=["cluster_centers_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but KMeans was fitted "
                f"with {int(self.n_features_in_)} features."
            )
        return torch.argmin(torch.cdist(Xt, self.cluster_centers_, p=2), dim=1)

    def fit_predict(self, X, y=None):
        """Fit and return training labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
            Cluster indices.
        """
        return self.fit(X, y).labels_
