"""Tests for torml metrics module.

This file contains tests for mean_squared_error, r2_score, and accuracy_score.
"""

import pytest
import torch
from torml.metrics import mean_squared_error, r2_score, accuracy_score


def test_mean_squared_error():
    """Test MSE calculation."""
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.5, 2.5, 3.5])
    # Expected MSE: ((0.5)^2 + (-0.5)^2 + (-0.5)^2) / 3 = (0.25 + 0.25 + 0.25) / 3 = 0.75 / 3 = 0.25
    assert mean_squared_error(y_true, y_pred) == pytest.approx(0.25)


def test_r2_score():
    """Test R2 score calculation."""
    # Perfect fit: MSE=0, R2=1.0
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    assert r2_score(y_true, y_pred) == pytest.approx(1.0)

    # Poor fit: MSE = variance of y_true, R2=0.0
    y_true = torch.tensor([1.0, 2.0, 3.0])
    # Mean of y_true is 2.0. If we predict the mean, MSE = variance.
    y_pred = torch.tensor([2.0, 2.0, 2.0])
    assert r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    # Worse than mean: MSE > variance, R2 < 0.0
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([0.0, 0.0, 0.0])
    assert r2_score(y_true, y_pred) < 0.0


def test_accuracy_score():
    """Test accuracy score calculation."""
    # Perfect classification: 2/2 correct
    y_true = torch.tensor([0.0, 1.0])
    y_pred = torch.tensor([0.0, 1.0])
    assert accuracy_score(y_true, y_pred) == pytest.approx(1.0)

    # Zero correct: 0/2 correct
    y_true = torch.tensor([0.0, 1.0])
    y_pred = torch.tensor([1.0, 0.0])
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.0)

    # Half correct: 1/2 correct
    y_true = torch.tensor([0.0, 1.0])
    y_pred = torch.tensor([0.0, 0.0])
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.5)
