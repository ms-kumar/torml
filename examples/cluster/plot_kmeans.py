"""Cluster blobs with KMeans and report inertia.

Run with ``python examples/cluster/plot_kmeans.py``.
"""

from __future__ import annotations

import torch

from torml.cluster import KMeans


def main() -> None:
    """Run the KMeans clustering example."""
    torch.manual_seed(0)
    X = torch.cat(
        [torch.randn(50, 2) + torch.tensor([-3.0, 0.0]), torch.randn(50, 2) + 3.0]
    )
    km = KMeans(n_clusters=2, random_state=0).fit(X)
    print(f"centers: {km.cluster_centers_.tolist()}")
    print(f"inertia: {km.inertia_:.2f}")


if __name__ == "__main__":
    main()
