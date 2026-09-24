"""Chain scaling and classification in a Pipeline.

Run with ``python examples/pipelines/plot_pipeline.py``.
"""

from __future__ import annotations

import torch

from torml.neighbors import KNeighborsClassifier
from torml.pipelines import Pipeline
from torml.preprocessing import StandardScaler


def main() -> None:
    """Run the pipeline example."""
    torch.manual_seed(0)
    x0 = torch.randn(50, 2) * 10 + torch.tensor([-30.0, 0.0])
    x1 = torch.randn(50, 2) * 10 + torch.tensor([30.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(3))]
    ).fit(X, y)
    print(f"train accuracy: {float(pipe.score(X, y)):.4f}")


if __name__ == "__main__":
    main()
