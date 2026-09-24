"""Multivariate-output meta-estimators with a PyTorch backend."""

from __future__ import annotations

from ._multioutput import MultiOutputClassifier, MultiOutputRegressor

__all__ = ["MultiOutputClassifier", "MultiOutputRegressor"]
