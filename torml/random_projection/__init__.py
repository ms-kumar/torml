"""Random projection with a PyTorch backend."""

from __future__ import annotations

from ._gaussian import GaussianRandomProjection, johnson_lindenstrauss_min_dim

__all__ = ["GaussianRandomProjection", "johnson_lindenstrauss_min_dim"]
