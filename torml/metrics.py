"""Metric functions.

Basic scoring functions for classification and regression.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

__all__ = ["mean_squared_error", "r2_score", "accuracy_score"]


def mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    squared: bool = True,
) -> torch.Tensor:
    """Compute mean squared error between y_true and y.

    Parameters
    ----------
    y_true : tensor of shape (n_samples,)
        The target values.
    y_pred : tensor of shape (n_samples,)
        The predicted values.
    sample_weight : tensor, optional
        Sample weights.
    squared : bool, default=True
        If True, return MSE (mean of squared errors).
        If False, return RMSE (standard deviation of errors).

    Returns
    -------
    float or tensor
        Mean squared error.
    """
    diff = y_true - y_pred
    if squared:
        return (diff ** 2).mean()
    else:
        return torch.sqrt((diff ** 2).mean())


def r2_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:
    """Compute R² score (coefficient of determination).

    R² = 1 - (sum of squared errors) / (sum of squared residuals)

    Where:
    - Sum of squared errors = sum((y_true - y_pred)^2)
    - Sum of squared residuals = sum((y_true - y_mean)^2)

    Parameters
    ----------
    y_true : tensor of shape (n_samples,)
        True/observed target values (ground truth).
    y_pred : tensor of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        R² score.

    Notes
    -----
    R² is an unbiased estimator when using OLS (Orthogonal Linear Spline) model
    with intercept. It is biased otherwise.

    In the latter case, using the convention of defining the total sum of
    residuals relative to the training set instead of the test set leads to:

    .. math::

        R^2 = 100\\cdot(1 - \\frac{MSE}{\\frac{1}{n}\\sum_{i=1}^n (y_i -\\overline{y})^2})

    where :math:`\\overline{y}` is the mean of :math:`y_true` and
    :math:`n` is the number of observations.

    The mean of the R² estimator over a cross-validation strategy approximates
    the R² estimator over the whole training set, but only if the whole
    training set is partitioned at least twice, see below.

    The score uses the residuals on the training set so that a negative R² can
    take values lower than :math:`-1` in general:

    .. math::

        R^2 = 100\\cdot(1 - \\frac{\\sum_{i=1}^n (y_{true,i} -y_{pred,i})^2}{
            \\sum_{i=1}^n (y_{true,i} -\\overline{y})^2)
    """
    y_pred = torch.as_tensor(y_pred)
    y_true = torch.as_tensor(y_true)

    y_true = torch.squeeze(y_true).float()
    y_pred = torch.squeeze(y_pred).float()

    if y_true.shape[0] == 1 and y_true.ndim == 2 and y_pred.ndim == 1:
        y_true = y_true.T

    y_pred = torch.squeeze(y_pred).float()

    y_true = torch.squeeze(y_true).float()

    if len(y_true.shape) == 1:
        y_pred = y_pred.unsqueeze(1)

    ss_res = (y_true - y_pred) ** 2
    ss_tot = (y_true - y_true.mean()) ** 2

    r2 = 1 - ss_res.mean() / ss_tot.mean()

    return r2


def accuracy_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    normalize: Literal["average", True] = "average",
    sample_weight: torch.Tensor | None = None,
    pos_label: int | None = None,
    label_binarize: bool = False,
    ignore_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute accuracy classification score.

    Accuracy :math:`\\frac{N_{correct}}{N_{total}}`

    This is a wrapper function that handles both single-label and
    multilabel input.

    Parameters
    ----------
    y_true : tensor of shape (n_samples,)
        True binary class labels.
    y_pred : tensor of shape (n_samples,) or (n_samples, n_labels)
        Predicted binary class labels.
    normalize : str, default="average"
        "average", or 3:
        Return the average accuracy, or the per-class accuracy vector.
    sample_weight : tensor, optional
        Sample weights for the accuracy computation.
    pos_label : int, optional
        Label that indicates the positive class, if y_true and y_pred tensors
        contain labels.
    ignore_indices : tensor, optional
        Indices to ignore when computing accuracy.

    Returns
    -------
    score : tensor
        Accuracy scores for each class if normalize="average" or
        single-value accuracy if otherwise.
    """
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)

    # Normalize both to torch tensors for consistent handling
    if y_true.shape[0] == 1 and y_true.ndim == 2:
        y_true = y_true.T
    if y_pred.shape[0] == 1 and y_pred.ndim == 2:
        y_pred = y_pred.T

    if label_binarize:
        y_true = y_true.to(dtype="float32")
        y_pred = y_pred.to(dtype="float32")

        # Apply threshold
        if normalize == "average":
            threshold = 0.5
        else:
            threshold = pos_label

        y_true = (y_true > threshold).int()
        y_pred = (y_pred > threshold).int()
    else:
        y_true = y_true.to(dtype=torch.long)
        y_pred = y_pred.to(dtype=torch.long)

    score = (y_pred == y_true).sum() / torch.tensor(len(y_true))

    return score
