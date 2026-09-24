"""Ensemble methods with a PyTorch backend."""

from __future__ import annotations

from ._bagging import BaggingClassifier, BaggingRegressor
from ._forest import RandomForestClassifier, RandomForestRegressor
from ._voting import VotingClassifier, VotingRegressor

__all__ = [
    "BaggingClassifier",
    "BaggingRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "VotingClassifier",
    "VotingRegressor",
]
