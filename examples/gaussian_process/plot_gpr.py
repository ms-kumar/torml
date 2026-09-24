"""Fit a Gaussian process to sine data and report error.

Run with ``python examples/gaussian_process/plot_gpr.py``.
"""

from __future__ import annotations

import torch

from torml.gaussian_process import GaussianProcessRegressor


def main() -> None:
    """Run the Gaussian process example."""
    torch.manual_seed(0)
    X = torch.linspace(-2, 2, 15).unsqueeze(1)
    y = torch.sin(X.squeeze(1))
    gpr = GaussianProcessRegressor().fit(X, y)
    mean, std = gpr.predict(X, return_std=True)
    print(f"max error: {float((mean - y).abs().max()):.2e}")
    print(f"mean std: {float(std.mean()):.2e}")


if __name__ == "__main__":
    main()
