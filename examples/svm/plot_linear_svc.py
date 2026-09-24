"""Separate blobs with LinearSVC and report accuracy.

Run with ``python examples/svm/plot_linear_svc.py``.
"""

from __future__ import annotations

import torch

from torml.metrics import accuracy_score
from torml.svm import LinearSVC


def main() -> None:
    """Run the linear SVM example."""
    torch.manual_seed(0)
    x0 = torch.randn(50, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(50, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    clf = LinearSVC(random_state=0).fit(X, y)
    print(f"train accuracy: {accuracy_score(y, clf.predict(X)):.4f}")


if __name__ == "__main__":
    main()
