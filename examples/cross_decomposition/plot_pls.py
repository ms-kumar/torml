"""Fit PLSRegression and report R2.

Run with ``python examples/cross_decomposition/plot_pls.py``.
"""

from __future__ import annotations

import torch

from torml.cross_decomposition import PLSRegression


def main() -> None:
    """Run the PLS regression example."""
    torch.manual_seed(0)
    X = torch.randn(60, 5)
    y = X[:, 0] * 2 - X[:, 1]
    pls = PLSRegression(n_components=2).fit(X, y)
    print(f"test R2: {float(pls.score(X, y)):.4f}")


if __name__ == "__main__":
    main()
