"""Base estimator module.

Provides the base class and mixins for all estimators, along with
testing utilities for estimator conformance checks.
"""

from __future__ import annotations

import inspect
import reprlib

import torch

from ._base import (
    BaseEstimator,
    ClassifierMixin,
    CloneMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
    is_classifier,
    is_clusterer,
    is_regressor,
)
from ._estimator_checks import check_estimator

__all__ = [
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "TransformerMixin",
    "ClusterMixin",
    "CloneMixin",
    "clone",
    "is_classifier",
    "is_regressor",
    "is_clusterer",
    "check_estimator",
]
