"""Estimate a covariance and score held-out data.

Run with ``python examples/covariance/plot_empirical.py``.
"""

from __future__ import annotations

import torch

from torml.covariance import EmpiricalCovariance


def main() -> None:
    """Run the empirical covariance example."""
    torch.manual_seed(0)
    X = torch.randn(100, 3)
    ec = EmpiricalCovariance().fit(X)
    print(f"location: {ec.location_.tolist()}")
    print(f"score: {ec.score(X):.4f}")


if __name__ == "__main__":
    main()
