"""Classify blobs with GaussianNB and report accuracy.

Run with ``python examples/naive_bayes/plot_gaussian.py``.
"""

from __future__ import annotations

import torch

from torml.metrics import accuracy_score
from torml.naive_bayes import GaussianNB


def main() -> None:
    """Run the GaussianNB classification example."""
    torch.manual_seed(4)
    x0 = torch.randn(50, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(50, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    clf = GaussianNB().fit(X, y)
    print(f"train accuracy: {accuracy_score(y, clf.predict(X)):.4f}")


if __name__ == "__main__":
    main()
