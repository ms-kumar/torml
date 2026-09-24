from __future__ import annotations

import pytest
import torch

from torml.linear_model import LinearRegression


@pytest.fixture(scope="class")
def fixture_data():
    """Provides data and expected values for regression testing."""
    # Simple linear relationship: y = 2*x1 + 3*x2 + 4
    X_train = torch.randn(50, 2)  # (n_samples, n_features=2)
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + 4
    X_test = torch.randn(5, 2)  # Prediction data
    return X_train, y_train, X_test


def test_linear_regression_fit_predict(fixture_data):
    """Tests the end-to-end fit and predict cycle."""
    X_train, y_train, X_test = fixture_data

    # Initialize and fit the model
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # Check that the model has learned coefficients
    assert hasattr(model, "coef_")
    # 2 features -> coef_ shape (2,), intercept_ shape (1,)
    assert tuple(model.coef_.shape) == (2,)
    assert hasattr(model, "intercept_")

    # Predict and check the output shape
    y_pred = model.predict(X_test)
    assert tuple(y_pred.shape) == (5,)

    # Coefficients should be close to [2, 3] with intercept ~4
    torch.testing.assert_close(
        model.coef_, torch.tensor([2.0, 3.0]), rtol=1e-3, atol=1e-3
    )


def test_linear_regression_predict_unfitted():
    """Ensures predict raises if fit() has not been called."""
    from torml.utils import NotFittedError

    model = LinearRegression()
    X_test = torch.randn(1, 2)
    with pytest.raises((NotFittedError, ValueError)):
        model.predict(X_test)


def test_linear_regression_fit_no_intercept():
    """Tests fitting without an intercept term."""
    X = torch.randn(10, 3)  # Use more samples for robustness
    y = X[:, 0] * 2 + X[:, 1] * 3  # Dependent on two features
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Should learn 3 coefficients (one for each feature)
    assert tuple(model.coef_.shape) == (3,)


def test_linear_regression_fit_feature_count():
    """Tests that the model correctly handles feature count."""
    X = torch.randn(5, 2)  # 2 features
    y = torch.randn(5)
    model = LinearRegression()
    model.fit(X, y)
    assert model.n_features_in_ == 2


def test_linear_regression_predict_dimension_mismatch():
    """Tests robust behavior when prediction inputs are dimensionally wrong."""
    # Train on 3 features, predict with 2
    X_train = torch.randn(10, 3)
    y_train = torch.randn(10)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Attempt to predict with data missing the 3rd feature
    X_predict_short = torch.randn(5, 2)
    with pytest.raises(ValueError):
        model.predict(X_predict_short)
