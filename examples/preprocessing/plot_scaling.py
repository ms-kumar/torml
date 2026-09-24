"""Scale features and encode labels.

Run with ``python examples/preprocessing/plot_scaling.py``.
"""

from __future__ import annotations

import torch

from torml.preprocessing import LabelEncoder, StandardScaler


def main() -> None:
    torch.manual_seed(2)
    X = torch.randn(10, 2) * 10 + 5
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    print(f"scaled mean: {Xs.mean(dim=0).tolist()}")
    print(f"round-trip ok: {bool(torch.allclose(scaler.inverse_transform(Xs), X))}")

    enc = LabelEncoder().fit(["cat", "dog", "cat", "bird"])
    print(f"classes: {list(enc.classes_)}")
    print(f"encoded: {enc.transform(['dog', 'bird']).tolist()}")


if __name__ == "__main__":
    main()
