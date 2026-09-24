"""Feature selection with a PyTorch backend."""

from __future__ import annotations

from ._univariate import SelectKBest, f_classif

__all__ = ["SelectKBest", "f_classif"]
