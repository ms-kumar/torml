"""Metrics module.

Provides scoring functions for evaluating model performance.
"""

from __future__ import annotations

import torch

__all__ = ["accuracy_score", "mean_squared_error", "r2_score"]


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute accuracy: fraction of correctly classified samples.

    Parameters
    ----------
    y_true : torch.Tensor of shape (n_samples,)
        Ground truth labels.
    y_pred : torch.Tensor of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    score : float
        Accuracy score in range [0, 1].
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Input shapes must match. Got {y_true.shape} and {y_pred.shape}."
        )

    # Convert tensors to numpy for comparison
    return_array = torch.asarray(y_true)
    pred_array = torch.asarray(y_pred)

    # Count correct predictions
    correct = (return_array == pred_array).sum().item()
    total = return_array.numel()

    return correct / total


def mean_squared_error(
    y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight=None
) -> float:
    """Compute mean squared error (MSE).

    Parameters
    ----------
    y_true : torch.Tensor of shape (n_samples,)
        Ground truth values.
    y_pred : torch.Tensor of shape (n_samples,)
        Predicted values.
    sample_weight : torch.Tensor or None, default=None
        Optional sample weights.

    Returns
    -------
    float
        Mean Squared Error of the predictions.

    Raises
    ------
    ValueError
        If input tensors are not comparable or have incompatible shapes.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Input shapes must match. Got {y_true.shape} and {y_pred.shape}."
        )

    # Calculate MSE: (1/N) * sum((y_true - y_pred)^2)
    mse = torch.mean((y_true - y_pred) ** 2).item()
    return mse


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Coefficient of determination (R-squared).

    Parameters
    ----------
    y_true : torch.Tensor of shape (n_samples,) or (n_samples, 1)
        Ground truth values.
    y_pred : torch.Tensor of shape (n_samples,) or (n_samples, 1)
        Predicted values.

    Returns
    -------
    float
        R-squared score of the model fit.

    Raises
    ------
    ValueError
        If input tensors are not comparable or have incompatible shapes.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Input shapes must match. Got {y_true.shape} and {y_pred.shape}."
        )

    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)

    # Calculate variance of true values (SS_total)
    ss_total = torch.sum((y_true - y_true.mean()) ** 2)

    # Handle case where variance is zero (all true values are the same)
    if ss_total.item() == 0:
        return 1.0 if mse == 0 else 0.0

    # Calculate R2: 1 - (SS_res / SS_tot)
    ss_res = mse * len(y_true)
    r2 = 1.0 - ss_res / ss_total.item()
    return float(r2)
