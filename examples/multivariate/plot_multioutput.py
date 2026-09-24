"""Fit two outputs at once with MultiOutputRegressor.

Run with ``python examples/multivariate/plot_multioutput.py``.
"""

from __future__ import annotations

import torch

from torml.linear_model import LinearRegression
from torml.multivariate import MultiOutputRegressor


def main() -> None:
    """Run the multi-output example."""
    torch.manual_seed(0)
    X = torch.randn(40, 3)
    Y = torch.stack([X[:, 0] * 2, X[:, 1] * -1], dim=1)
    pred = MultiOutputRegressor(LinearRegression()).fit(X, Y).predict(X)
    print(f"shape: {tuple(pred.shape)}")
    print(f"max error: {float((pred - Y).abs().max()):.2e}")


if __name__ == "__main__":
    main()
