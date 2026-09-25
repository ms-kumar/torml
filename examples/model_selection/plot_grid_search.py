"""Tune k with GridSearchCV and report the best params.

Run with ``python examples/model_selection/plot_grid_search.py``.
"""

from __future__ import annotations

import torch

from torml.model_selection import GridSearchCV
from torml.neighbors import KNeighborsClassifier


def main() -> None:
    """Run the grid search example."""
    torch.manual_seed(0)
    x0 = torch.randn(30, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(30, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(30), torch.ones(30)]).long()

    gs = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [1, 3, 7]}, cv=3)
    gs.fit(X, y)
    print(f"best params: {gs.best_params_}")
    print(f"best score: {gs.best_score_:.4f}")


if __name__ == "__main__":
    main()
