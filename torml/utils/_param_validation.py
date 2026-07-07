"""Parameter validation utilities.

Validation functions for estimator parameters.
"""

from __future__ import annotations

import inspect
from typing import Literal, Sequence


class InvalidParameterError(ValueError):
    """Raised when a parameter does not meet validation constraints."""

    pass


class Interval:
    """Interval for parameter validation.

    Parameters
    ----------
    low : float or None
        Lower bound (inclusive if low_closed=True).
    high : float or None
        Upper bound (inclusive if high_closed=True).
    low_closed : bool, default=False
        Whether low bound is inclusive.
    high_closed : bool, default=False
        Whether high bound is inclusive.
    """

    def __init__(
        self,
        low: float | None = None,
        high: float | None = None,
        low_closed: bool = False,
        high_closed: bool = False,
    ):
        self.low = low
        self.high = high
        self.low_closed = low_closed
        self.high_closed = high_closed

    def __call__(self, x) -> bool:
        """Check if x is in the interval."""
        if self.low is not None and self.high is not None:
            if not self.low_closed and x == self.low:
                return False
            if not self.high_closed and x == self.high:
                return False
            if x < self.low or x > self.high:
                return False
        elif self.low is not None and self.low_closed:
            if x < self.low:
                return False
        elif self.high is not None and self.high_closed:
            if x > self.high:
                return False
        return True

    def __repr__(self):
        if self.low is None and self.high is None:
            return "(open interval)"
        if self.low is not None and self.high is not None:
            if self.low_closed and self.high_closed:
                return f"[{self.low}, {self.high}]"
            if not self.low_closed and not self.high_closed:
                return f"({self.low}, {self.high})"
            if self.low_closed:
                return f"[{self.low}, {self.high})"
            if self.high_closed:
                return f"({self.low}, {self.high}]"
        if self.low is not None and self.low_closed:
            return f"[{self.low}, +inf)"
        if self.high is not None and self.high_closed:
            return f"(-inf, {self.high}]"
        if self.low is not None and self.high is None:
            return f"[{self.low}, +inf)"
        if self.low is None and self.high is not None:
            return f"(-inf, {self.high}]"
        return "(-inf, +inf)"


class StrOptions:
    """String options for parameter validation.

    Parameters
    ----------
    valid_strings : list or set of str
        Valid string values for the parameter.
    case_sensitive : bool, default=True
        Whether to match case-sensitively.
    """

    def __init__(
        self,
        valid_strings: Sequence | set,
        case_sensitive: bool = True,
    ):
        if not isinstance(valid_strings, str):
            valid_strings = set(valid_strings)
        else:
            valid_strings = {valid_strings}

        self.valid_strings = valid_strings
        self.case_sensitive = case_sensitive

    def __call__(self, x) -> bool:
        """Check if x is in the valid strings."""
        if self.case_sensitive:
            return x in self.valid_strings
        else:
            return str(x).lower() in {s.lower() for s in self.valid_strings}

    def __repr__(self):
        return f'{repr(self.valid_strings)}'


class HasMethods:
    """Check if an estimator has specific methods.

    Parameters
    ----------
    names : list of str
        Method names that must be present on the estimator.
    """

    def __init__(self, names: Sequence[str]):
        self.names = list(names)

    def __call__(self, estimator) -> bool:
        """Check if all required methods exist."""
        for name in self.names:
            if not hasattr(estimator, name):
                return False
        return True

    def __repr__(self):
        return f"{self.names}"


def validate_parameter_constraints(
    parameter_name: str,
    constraints,
    original_params: dict,
    *,
    allow_unknown_params: bool = False,
    invalid_name_message: str | None = None,
):
    """Validate a single parameter against given constraints.

    Parameters
    ----------
    parameter_name : str
        Name of the parameter to validate.
    constraints
        Constraints to apply. Can be validators, Interval, StrOptions, HasMethods,
        custom callable, or tuple for custom validation with message.
    original_params : dict
        Original parameters dict (before filtering).
    allow_unknown_params : bool, default=False
        If False and parameter is not in original_params, raise error.
    invalid_name_message : str, optional
        Custom message for invalid parameter name.

    Raises
    ------
    InvalidParameterError
        If parameter fails validation.
    TypeError
        If parameter is not provided.
    """
    original_param = original_params.get(parameter_name)
    is_original_param = parameter_name in original_params

    if original_param is None:
        if is_original_param:
            raise InvalidParameterError(
                f"The {parameter_name!r} parameter is not specified."
            )
        elif not allow_unknown_params:
            if invalid_name_message is not None:
                raise InvalidParameterError(invalid_name_message)
            raise InvalidParameterError(
                f"The {parameter_name!r} parameter is not specified."
            )

        constraints = []

    for validation_object in constraints:
        if validation_object is None:
            continue
        elif isinstance(constraints[0], Interval):
            if not validation_object(original_param):
                return validation_object(original_param)
        elif isinstance(validation_object, StrOptions):
            if not validation_object(original_param):
                raise InvalidParameterError(
                    f"The {parameter_name!r} parameter value for the "
                    f"{validation_object} validator should be one of "
                    f"{validation_object}, got {original_param!r}."
                )
        elif isinstance(validation_object, HasMethods):
            if not validation_object(original_param):
                raise InvalidParameterError(
                    f"The {parameter_name!r} parameter must have methods "
                    f"'{validation_object.names}', got {validation_object}'."
                )
        elif callable(validation_object):
            if not (validation_object(original_param)):
                raise InvalidParameterError(
                    f"The {parameter_name!r} parameter does not pass the "
                    f"{validation_object} validator."
                )

    if parameter_name not in original_params:
        if is_original_param:
            raise InvalidParameterError(
                f"The {parameter_name!r} parameter is not specified."
            )
        else:
            constraints = []

    return parameter_name, original_param
