"""Project data with GaussianRandomProjection.

Run with ``python examples/random_projection/plot_grp.py``.
"""

from __future__ import annotations

import torch

from torml.random_projection import GaussianRandomProjection


def main() -> None:
    """Run the random projection example."""
    torch.manual_seed(0)
    X = torch.randn(50, 10)
    grp = GaussianRandomProjection(n_components=4, random_state=0).fit(X)
    print(f"shape: {tuple(grp.transform(X).shape)}")


if __name__ == "__main__":
    main()
