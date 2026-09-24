"""Classify blobs with KNeighborsClassifier and report accuracy.

Run with ``python examples/neighbors/plot_kneighbors.py``.
"""

from __future__ import annotations

import torch

from torml.metrics import accuracy_score
from torml.neighbors import KNeighborsClassifier


def main() -> None:
    """Run the k-neighbors classification example."""
    torch.manual_seed(3)
    x0 = torch.randn(50, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(50, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    pred = clf.predict(X)
    print(f"train accuracy: {accuracy_score(y, pred):.4f}")


if __name__ == "__main__":
    main()
