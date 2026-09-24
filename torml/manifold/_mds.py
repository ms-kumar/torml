"""Classical multidimensional scaling with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator
from torml.utils._validation import check_array, check_is_fitted


class MDS(BaseEstimator):
    """Classical MDS via double centering and eigendecomposition.

    Parameters
    ----------
    n_components : int, default=2
        Embedding dimensionality.
    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Whether ``fit`` receives data or a dissimilarity matrix.

    Attributes
    ----------
    embedding_ : torch.Tensor of shape (n_samples, n_components)
        Learned positions.
    stress_ : float
        Normalized reconstruction error of inner products.
    n_features_in_ : int
        Number of features seen during fit (data input only).
    """

    name = "MDS"

    def __init__(self, n_components=2, dissimilarity="euclidean"):
        self.n_components = n_components
        self.dissimilarity = dissimilarity

    def fit(self, X, y=None):
        """Embed ``X`` to preserve pairwise distances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Data, or dissimilarity matrix when ``dissimilarity='precomputed'``.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : MDS
            Fitted embedder.
        """
        if isinstance(self.n_components, bool) or not isinstance(
            self.n_components, int
        ):
            raise TypeError(
                f"n_components must be an int, got {type(self.n_components).__name__}."
            )
        if int(self.n_components) < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if self.dissimilarity not in ("euclidean", "precomputed"):
            raise ValueError(
                "dissimilarity must be 'euclidean' or 'precomputed', "
                f"got {self.dissimilarity!r}."
            )
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        n_samples = int(Xt.shape[0])
        if int(self.n_components) > n_samples:
            raise ValueError(
                f"n_components ({self.n_components}) must be "
                f"<= n_samples ({n_samples})."
            )
        if self.dissimilarity == "euclidean":
            self.n_features_in_ = int(Xt.shape[1])
            dist = torch.cdist(Xt, Xt, p=2)
        else:
            if int(Xt.shape[0]) != int(Xt.shape[1]):
                raise ValueError("Precomputed dissimilarity must be square.")
            dist = Xt
        gram = -0.5 * (dist**2)
        gram = (
            gram
            - gram.mean(dim=0, keepdim=True)
            - gram.mean(dim=1, keepdim=True)
            + gram.mean()
        )
        eigenvals, eigenvecs = torch.linalg.eigh(gram)  # pylint: disable=not-callable
        order = torch.argsort(eigenvals, descending=True)
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
        k = int(self.n_components)
        positive = eigenvals[:k].clamp(min=0)
        self.embedding_ = eigenvecs[:, :k] * torch.sqrt(positive).unsqueeze(0)
        recon = self.embedding_ @ self.embedding_.T
        denom = float((gram**2).sum())
        self.stress_ = float(((gram - recon) ** 2).sum() / denom) if denom > 0 else 0.0
        return self

    def fit_transform(self, X, y=None):
        """Fit and return the embedding.

        Parameters
        ----------
        X : array-like
            Data or dissimilarity matrix.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        embedding : torch.Tensor of shape (n_samples, n_components)
            Learned positions.
        """
        return self.fit(X, y).embedding_

    def score(self, X, y=None):
        """Return the negative stress (higher is better).

        Parameters
        ----------
        X : array-like
            Data or dissimilarity matrix (same form as fit).
        y : ignored
            Present for API compatibility.

        Returns
        -------
        score : float
            Negative stress of a fresh embedding.
        """
        check_is_fitted(self, attributes=["embedding_"])
        return (
            -type(self)(
                n_components=int(self.n_components), dissimilarity=self.dissimilarity
            )
            .fit(X)
            .stress_
        )
