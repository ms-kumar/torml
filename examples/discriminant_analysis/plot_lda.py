"""Classify blobs with LDA and report accuracy.

Run with ``python examples/discriminant_analysis/plot_lda.py``.
"""

from __future__ import annotations

import torch

from torml.discriminant_analysis import LinearDiscriminantAnalysis
from torml.metrics import accuracy_score


def main() -> None:
    """Run the LDA example."""
    torch.manual_seed(0)
    X = torch.cat(
        [torch.randn(40, 2) + torch.tensor([-3.0, 0.0]), torch.randn(40, 2) + 3.0]
    )
    y = torch.cat([torch.zeros(40), torch.ones(40)]).long()
    clf = LinearDiscriminantAnalysis().fit(X, y)
    print(f"train accuracy: {accuracy_score(y, clf.predict(X)):.4f}")


if __name__ == "__main__":
    main()
