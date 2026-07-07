"""Validation utilities.

Shared validation functions for checking inputs and fitted state.
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import numpy
import torch


class NotFittedError(ValueError):
    """Error raised when calling methods on unfitted estimators.

    Subclass of :class:`ValueError`.
    """

    pass


def check_array(
    array,
    *,
    dtype: Literal["float32", "float64", "int32", "int64"] = torch.float32,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    copy: bool = False,
    force_all_finite: bool = True,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
    device: torch.device | None = None,
    input_name: str = "X",
) -> torch.Tensor:
    """Check that ``array`` is properly formatted as an input (X or y).

    Parameters
    ----------
    array : array-like of shape (n_samples, n_features) or (n_samples,)
        Data to check. Can be a torch.Tensor, numpy.ndarray, or list.
    dtype : data-type, default=torch.float32
        Data type for the output tensor.
    ensure_2d : bool, default=True
        If True, X will be converted to 2D array. If False, 1D arrays are allowed.
    allow_nd : bool, default=False
        If True, higher-dimensional arrays are allowed.
    copy : bool, default=False
        If True, ensure that output is a copy of the input.
    force_all_finite : bool, default=True
        If True, raise ValueError if X contains NaN or Inf.
    ensure_min_samples : int, default=1
        Minimum number of samples required.
    ensure_min_features : int, default=1
        Minimum number of features required.
    device : torch.device or None, optional
        Device to put the output tensor on. If None, preserve caller's device.
    input_name : str, default="X"
        Name of the input for error messages.

    Returns
    -------
    output : torch.Tensor
        Checked and optionally converted tensor.

    Raises
    ------
    TypeError
        If ``array`` cannot be converted to a tensor.
    ValueError
        If ``array`` does not pass validation (wrong shape, contains NaN/Inf, etc.).
    """
    # Convert to tensor
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, numpy.ndarray):
        tensor = torch.as_tensor(numpy.asarray(array), dtype=dtype)
    else:
        try:
            tensor = torch.as_tensor(array, dtype=dtype)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"{input_name} must be an array-like or torch.Tensor, "
                f"not {type(array).__name__}."
            ) from e

    # Check device
    if device is not None:
        tensor = tensor.to(device)

    # Check for 2D or correct shape
    if ensure_2d:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim > 2:
            raise ValueError(
                f"{input_name} should be a 1D or 2D array, got {tensor.ndim}D array."
            )
    elif not allow_nd and tensor.ndim > 2:
        raise ValueError(
            f"{input_name} should be a 1D or 2D array, got {tensor.ndim}D array."
        )

    # Check min samples/features
    if tensor.shape[0] < ensure_min_samples:
        raise ValueError(
            f"{input_name} must have at least {ensure_min_samples} samples, "
            f"got {tensor.shape[0]}."
        )

    if tensor.ndim == 2 and tensor.shape[1] < ensure_min_features:
        raise ValueError(
            f"{input_name} must have at least {ensure_min_features} features, "
            f"got {tensor.shape[1]}."
        )

    # Check for all-finite
    if force_all_finite:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(
                f"{input_name} contains NaN or infinite values."
            )

    # Return copy if needed
    if copy and tensor.device.type == "cpu" and hasattr(array, "copy"):
        tensor = tensor.clone()

    return tensor


def check_X_y(
    X,
    y,
    *,
    accept_sparse=False,
    ensure_all_finite=True,
    input_name_X="X",
    input_name_y="y",
):
    """Concatenate two sets of 1D/2D inputs.

    The main difference between this function and check_array is this
    one accepts 1D inputs. Also, the dtype and ensure_2d checks are
    performed only for X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    Xt, y_t : tuple of (np.ndarray, np.ndarray) or torch.Tensor, torch.Tensor
        Processed arrays.
    """
    X_tensor = check_array(X, ensure_2d=True, accept_sparse=accept_sparse)
    y_tensor = check_array(
        y, ensure_2d=False, accept_sparse=accept_sparse, input_name=input_name_y
    )

    if X_tensor.shape[0] != y_tensor.shape[0]:
        raise ValueError(
            f"The X {input_name_X} and y {input_name_y} parameters have different "
            f"lengths ({X_tensor.shape[0]} in X, {y_tensor.shape[0]} in y)."
        )

    return X_tensor, y_tensor


def column_or_1d(y: Sequence) -> torch.Tensor:
    """Ensure y is 1D, convert it to an array.

    Parameters
    ----------
    y : array-like or list of shape (n_samples,) or [n_samples, n_outputs]
        Target values.

    Returns
    -------
    y : torch.Tensor
        Converted array, guaranteed to be 1D.
    """
    y_tensor = check_array(
        y, ensure_2d=True, accept_sparse=False, dtype=torch.float32, input_name="y"
    )

    if y_tensor.shape[0] == 1 and y_tensor.ndim == 2:
        y_tensor = y_tensor.squeeze(0)

    return y_tensor


def check_is_fitted(estimator, attributes=None, *, msg: str | None = None):
    """Raise an error if estimator is not fitted or if required attributes are not set.

    Parameters
    ----------
    estimator : object
        Estimator instance.
    attributes : list of tuples, optional
        List of (name, is_fitted) pairs to check.
    msg : str, optional
        Custom error message.

    Returns
    -------
    estimator : object
        The same estimator instance passed as input.

    Raises
    ------
    NotFittedError
        If the estimator is not fitted.
    """
    if attributes is None:
        fitted_attributes = ("coefs", "labels_", "labels_", "coef_", "intercept_")
    else:
        fitted_attributes = attributes

    for name, _ in fitted_attributes:
        if not hasattr(estimator, name):
            msg_to_show = f"The {name!r} attribute of the object " "was not found."
            if msg is None:
                msg = msg_to_show
            else:
                msg = msg + "\n" + msg_to_show

            raise NotFittedError(msg)

    return estimator


def check_scalar(
    x,
    name: str,
    target_type: Sequence[str] | str,
    *,
    min_val: int | float | None = None,
    max_val: int | float | None = None,
    include_boundaries="both",
) -> any:
    """Check that x is a scalar with a given type and bounds.

    Parameters
    ----------
    x : int, float, bool, or None
        Value to check.
    name : str
        Name of the scalar for error messages.
    target_type : list of str or str, default=["int", "float", "bool"]
        Valid target types for x.
    min_val : int or float or None, default=None
        Lower bound on the value.
    max_val : int or float or None, default=None
        Upper bound on the value.
    include_boundaries : str, default="both"
        Whether to include min and max in valid range. Can also be "lower",
        "upper", or "none".

    Returns
    -------
    x : scalar
        The input value if it passes checks.

    Raises
    ------
    TypeError
        If x is not one of the target types or is None (if all targets are int/float).
    ValueError
        If x is outside the valid bounds.
    """
    if not isinstance(target_type, Sequence):
        target_type = [target_type]
    elif len(target_type) < 2 and target_type[0] == "str":
        target_type = target_type[0]

    if not isinstance(x, target_type):
        if target_type == "int":
            raise TypeError("An integer is required")
        elif target_type == "float":
            raise TypeError("A float is required")
        elif target_type == "bool":
            raise TypeError("A boolean is required")
        else:
            raise TypeError(f"The {name} parameter type must be one of {target_type}")
    elif x is None:
        if target_type not in ["int", "float", "bool"]:
            return x

    if isinstance(x, torch.Tensor):
        x_value = float(x.item())
    else:
        x_value = float(x)

    if min_val is not None and x_value < min_val:
        if "lower" not in include_boundaries:
            raise ValueError(
                f"The {name!r} parameter value must be greater than "
                f">= {min_val}. Got {x_value}."
            )
    elif max_val is not None and x_value > max_val:
        if "upper" not in include_boundaries:
            raise ValueError(
                f"The {name!r} parameter value must be in the range "
                f"[{min_val}, {max_val}] in {include_boundaries}. "
                f"Got {x_value}."
            )

    # Convert to target type
    if target_type == "int":
        return int(x)
    elif target_type == "float":
        return float(x)
    elif target_type == "bool":
        return bool(x)
    else:
        return x


def has_fit_parameter(estimator, parameter: str) -> bool:
    """Check if an estimator has a fit parameter with a given name.

    Parameters
    ----------
    estimator : object
        Estimator instance.
    parameter : str
        Name of the parameter to check.

    Returns
    -------
    has_parameter : bool
        True if estimator has the given fit parameter.
    """
    import inspect

    parameters = [
        p
        for p, v in inspect.signature(estimator.fit).parameters.items()
        if p != "self"
    ]

    return parameter in parameters
