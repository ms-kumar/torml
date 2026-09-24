"""Reduce blobs to 2D with PCA and report variance.

Run with ``python examples/decomposition/plot_pca.py``.
"""

from __future__ import annotations

import torch

from torml.decomposition import PCA


def main() -> None:
    """Run the PCA example."""
    torch.manual_seed(0)
    X = torch.randn(100, 5)
    pca = PCA(n_components=2).fit(X)
    print(f"explained ratio: {pca.explained_variance_ratio_.tolist()}")
    print(f"projected shape: {tuple(pca.transform(X).shape)}")


if __name__ == "__main__":
    main()
