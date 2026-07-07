"""Base estimator module.

Provides the base class and mixins for all estimators, along with
testing utilities for estimator conformance checks.
"""

from __future__ import annotations

import inspect
import reprlib

import numpy as np
import torch

from ._base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, ClusterMixin, clone, is_classifier, is_regressor, is_clusterer, CloneMixin
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
