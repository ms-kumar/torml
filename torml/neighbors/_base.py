"""Shared neighbor search utilities with a PyTorch backend."""

from __future__ import annotations

import torch


def _validate_weights(weights: str) -> str:
    """Validate the ``weights`` hyperparameter."""
    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}.")
    return weights


def _validate_p(p) -> float:
    """Validate the Minkowski ``p`` hyperparameter."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise TypeError(f"p must be a number >= 1 or inf, got {type(p).__name__}.")
    p = float(p)
    if p != float("inf") and p < 1:
        raise ValueError(f"p must be >= 1 or inf, got {p}.")
    return p


def _validate_n_neighbors(n_neighbors) -> int:
    """Validate the ``n_neighbors`` hyperparameter."""
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
        raise TypeError(
            f"n_neighbors must be an int, got {type(n_neighbors).__name__}."
        )
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")
    return n_neighbors


def pairwise_distances(
    X: torch.Tensor, Y: torch.Tensor, p: float = 2.0
) -> torch.Tensor:
    """Compute Minkowski distances between rows of ``X`` and ``Y``.

    Parameters
    ----------
    X : torch.Tensor of shape (n_queries, n_features)
        Query points.
    Y : torch.Tensor of shape (n_indexed, n_features)
        Indexed points.
    p : float, default=2.0
        Minkowski order. ``inf`` gives Chebyshev distance.

    Returns
    -------
    dist : torch.Tensor of shape (n_queries, n_indexed)
        Pairwise distances.
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y have different feature counts: {X.shape[1]} vs {Y.shape[1]}."
        )
    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    if p == float("inf"):
        return diff.abs().max(dim=-1).values
    return diff.abs().pow(p).sum(dim=-1).pow(1.0 / p)


def kneighbors(
    X_indexed: torch.Tensor,
    X_query: torch.Tensor,
    n_neighbors: int,
    p: float = 2.0,
):
    """Find ``n_neighbors`` nearest indexed points for each query.

    Parameters
    ----------
    X_indexed : torch.Tensor of shape (n_indexed, n_features)
        Training points.
    X_query : torch.Tensor of shape (n_queries, n_features)
        Query points.
    n_neighbors : int
        Number of neighbors, ``1 <= n_neighbors <= n_indexed``.
    p : float, default=2.0
        Minkowski order.

    Returns
    -------
    dist : torch.Tensor of shape (n_queries, n_neighbors)
        Distances to the selected neighbors.
    ind : torch.Tensor of shape (n_queries, n_neighbors)
        Indices into ``X_indexed``.
    """
    n_indexed = int(X_indexed.shape[0])
    if not 1 <= n_neighbors <= n_indexed:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must satisfy "
            f"1 <= n_neighbors <= n_samples ({n_indexed})."
        )
    dist_all = pairwise_distances(X_query, X_indexed, p=p)
    dist, ind = torch.topk(dist_all, k=n_neighbors, dim=1, largest=False, sorted=True)
    return dist, ind


def distance_weights(dist: torch.Tensor) -> torch.Tensor:
    """Convert distances to inverse-distance weights.

    Zero distances are handled by the caller (exact matches take over).
    """
    return 1.0 / (dist + 1e-12)
