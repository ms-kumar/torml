"""Fit a 2-component mixture and report weights.

Run with ``python examples/mixture/plot_gmm.py``.
"""

from __future__ import annotations

import torch

from torml.mixture import GaussianMixture


def main() -> None:
    """Run the Gaussian mixture example."""
    torch.manual_seed(0)
    X = torch.cat(
        [torch.randn(50, 2) + torch.tensor([-3.0, 0.0]), torch.randn(50, 2) + 3.0]
    )
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    print(f"weights: {gm.weights_.tolist()}")
    print(f"score: {gm.score(X):.4f}")


if __name__ == "__main__":
    main()
