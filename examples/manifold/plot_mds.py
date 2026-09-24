"""Embed data in 2D with MDS and report stress.

Run with ``python examples/manifold/plot_mds.py``.
"""

from __future__ import annotations

import torch

from torml.manifold import MDS


def main() -> None:
    """Run the MDS example."""
    torch.manual_seed(0)
    X = torch.randn(30, 4)
    mds = MDS(n_components=2).fit(X)
    print(f"embedding shape: {tuple(mds.embedding_.shape)}")
    print(f"stress: {mds.stress_:.4f}")


if __name__ == "__main__":
    main()
