"""Select the best features with SelectKBest.

Run with ``python examples/feature_selection/plot_kbest.py``.
"""

from __future__ import annotations

import torch

from torml.feature_selection import SelectKBest


def main() -> None:
    """Run the feature selection example."""
    torch.manual_seed(0)
    y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    X = torch.cat(
        [y.float().unsqueeze(1) * 3 + 0.5 * torch.randn(60, 2), torch.randn(60, 3)],
        dim=1,
    )
    sel = SelectKBest(k=2).fit(X, y)
    print(f"support: {sel.get_support(indices=True).tolist()}")
    print(f"shape: {tuple(sel.transform(X).shape)}")


if __name__ == "__main__":
    main()
