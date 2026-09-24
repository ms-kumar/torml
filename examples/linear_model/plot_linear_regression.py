"""Fit LinearRegression on synthetic data and report R2.

Run with ``python examples/linear_model/plot_linear_regression.py``.
"""

from __future__ import annotations

import torch

from torml.linear_model import LinearRegression
from torml.metrics import r2_score
from torml.model_selection import train_test_split


def main() -> None:
    torch.manual_seed(0)
    X = torch.randn(200, 3)
    true_coef = torch.tensor([1.0, 2.0, -1.0])
    y = X @ true_coef + 0.5 + 0.1 * torch.randn(200)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    print(f"coef: {model.coef_.tolist()}")
    print(f"intercept: {model.intercept_.tolist()}")
    print(f"test R2: {r2_score(y_test, model.predict(X_test)):.4f}")


if __name__ == "__main__":
    main()
