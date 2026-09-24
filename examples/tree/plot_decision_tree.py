"""Fit a decision tree on blobs and report accuracy.

Run with ``python examples/tree/plot_decision_tree.py``.
"""

from __future__ import annotations

import torch

from torml.metrics import accuracy_score
from torml.tree import DecisionTreeClassifier


def main() -> None:
    """Run the decision tree example."""
    torch.manual_seed(0)
    x0 = torch.randn(50, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(50, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    print(f"train accuracy: {accuracy_score(y, clf.predict(X)):.4f}")
    print(f"importances: {clf.feature_importances_.tolist()}")


if __name__ == "__main__":
    main()
