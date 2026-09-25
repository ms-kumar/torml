"""DBSCAN clustering with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import ClusterMixin
from torml.utils._validation import check_array


class DBSCAN(ClusterMixin):
    """Density-based clustering with noise label -1.

    Parameters
    ----------
    eps : float, default=0.5
        Neighborhood radius.
    min_samples : int, default=5
        Points required in a neighborhood for a core point.

    Attributes
    ----------
    labels_ : torch.Tensor of shape (n_samples,)
        Cluster labels (-1 for noise).
    core_sample_indices_ : torch.Tensor
        Indices of core samples.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "DBSCAN"

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def _expand_cluster(
        self, point, neighborhoods, is_core, labels, visited, cluster_id
    ):
        """Grow ``cluster_id`` from ``point`` over density-reachable samples."""
        labels[point] = cluster_id
        seeds = [j for j in neighborhoods[point] if j != point]
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if not bool(visited[j]):
                visited[j] = True
                if bool(is_core[j]):
                    for nb in neighborhoods[j]:
                        if nb not in seeds:
                            seeds.append(nb)
            if int(labels[j].item()) == -1:
                labels[j] = cluster_id
            k += 1

    def fit(self, X, y=None):
        """Find dense clusters in ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : DBSCAN
            Fitted clusterer.
        """
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)):
            raise TypeError(f"eps must be a float, got {type(self.eps).__name__}.")
        if float(self.eps) <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}.")
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise TypeError(
                "min_samples must be an int, " f"got {type(self.min_samples).__name__}."
            )
        if int(self.min_samples) < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}.")
        Xt = check_array(X, ensure_2d=True)
        n_samples = int(Xt.shape[0])
        self.n_features_in_ = int(Xt.shape[1])

        dist = torch.cdist(Xt, Xt, p=2)
        neighborhoods = [
            torch.where(dist[i] <= float(self.eps))[0].tolist()
            for i in range(n_samples)
        ]
        is_core = torch.tensor(
            [len(nb) >= int(self.min_samples) for nb in neighborhoods],
            dtype=torch.bool,
            device=Xt.device,
        )
        labels = torch.full((n_samples,), -1, dtype=torch.long, device=Xt.device)
        visited = torch.zeros(n_samples, dtype=torch.bool, device=Xt.device)
        cluster_id = 0
        for i in range(n_samples):
            if bool(visited[i]) or not bool(is_core[i]):
                continue
            visited[i] = True
            self._expand_cluster(i, neighborhoods, is_core, labels, visited, cluster_id)
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = torch.where(is_core)[0]
        return self

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
            Cluster labels (-1 for noise).
        """
        check_array(X, ensure_2d=True)
        return self.fit(X, y).labels_
