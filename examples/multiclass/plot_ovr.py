"""Classify 3 classes with OneVsRestClassifier.

Run with ``python examples/multiclass/plot_ovr.py``.
"""

from __future__ import annotations

import torch

from torml.metrics import accuracy_score
from torml.multiclass import OneVsRestClassifier
from torml.tree import DecisionTreeClassifier


def main() -> None:
    """Run the one-vs-rest example."""
    torch.manual_seed(0)
    X = torch.cat(
        [
            torch.randn(30, 2) + c
            for c in [
                torch.tensor([-3.0, 0.0]),
                torch.tensor([3.0, 0.0]),
                torch.tensor([0.0, 3.0]),
            ]
        ]
    )
    y = torch.cat([torch.full((30,), i) for i in range(3)])
    clf = OneVsRestClassifier(DecisionTreeClassifier()).fit(X, y)
    print(f"train accuracy: {accuracy_score(y, clf.predict(X)):.4f}")


if __name__ == "__main__":
    main()
