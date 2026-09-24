"""Cross-validate LinearRegression and report per-fold scores.

Run with ``python examples/model_selection/plot_cross_validation.py``.
"""

from __future__ import annotations

import torch

from torml.linear_model import LinearRegression
from torml.model_selection import cross_val_score


def main() -> None:
    """Run the cross-validation example."""
    torch.manual_seed(1)
    X = torch.randn(100, 2)
    y = 2 * X[:, 0] - X[:, 1] + 0.1 * torch.randn(100)

    scores = cross_val_score(LinearRegression(), X, y, cv=5)
    print(f"fold R2: {scores.tolist()}")
    print(f"mean R2: {float(scores.mean()):.4f}")


if __name__ == "__main__":
    main()
