"""Utility functions for the torml library.

This module provides shared utility functions and exceptions used throughout
the library, including input validation, random state management, and
estimator checks for testing.
"""

from __future__ import annotations

from ._validation import (
    NotFittedError,
    check_array,
    check_X_y,
    check_is_fitted,
    check_scalar,
    column_or_1d,
)

from ._param_validation import (
    HasMethods,
    Interval,
    InvalidParameterError,
    StrOptions,
    validate_parameter_constraints,
)

from ._random import RandomState, check_random_state

from ._mask import indices_to_mask, safe_mask

from ._tags import _DEFAULT_TAGS, get_tags

from ._estimator_checks import check_estimator

__all__ = [
    "check_array",
    "check_X_y",
    "check_is_fitted",
    "check_scalar",
    "check_random_state",
    "check_estimator",
    "column_or_1d",
    "get_tags",
    "HasMethods",
    "Interval",
    "InvalidParameterError",
    "NotFittedError",
    "RandomState",
    "StrOptions",
    "indices_to_mask",
    "safe_mask",
]
