"""Nearest-neighbor estimators with a PyTorch backend."""

from __future__ import annotations

from ._classification import KNeighborsClassifier
from ._regression import KNeighborsRegressor

__all__ = ["KNeighborsClassifier", "KNeighborsRegressor"]
