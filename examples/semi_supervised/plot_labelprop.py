"""Propagate partial labels and report recovery.

Run with ``python examples/semi_supervised/plot_labelprop.py``.
"""

from __future__ import annotations

import torch

from torml.semi_supervised import LabelPropagation


def main() -> None:
    """Run the label propagation example."""
    torch.manual_seed(0)
    X = torch.cat(
        [torch.randn(20, 2) + torch.tensor([-3.0, 0.0]), torch.randn(20, 2) + 3.0]
    )
    full = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    y = full.clone()
    y[5:15] = -1
    y[25:35] = -1
    lp = LabelPropagation().fit(X, y)
    print(f"recovery: {float((lp.predict(X) == full).float().mean()):.4f}")


if __name__ == "__main__":
    main()
