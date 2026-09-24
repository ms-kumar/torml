"""Combine trees and neighbors with soft voting.

Run with ``python examples/ensemble/plot_voting.py``.
"""

from __future__ import annotations

import torch

from torml.ensemble import VotingClassifier
from torml.metrics import accuracy_score
from torml.neighbors import KNeighborsClassifier
from torml.tree import DecisionTreeClassifier


def main() -> None:
    """Run the voting ensemble example."""
    torch.manual_seed(0)
    x0 = torch.randn(50, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(50, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    clf = VotingClassifier(
        [("tree", DecisionTreeClassifier()), ("knn", KNeighborsClassifier(3))],
        voting="soft",
    ).fit(X, y)
    print(f"train accuracy: {accuracy_score(y, clf.predict(X)):.4f}")


if __name__ == "__main__":
    main()
