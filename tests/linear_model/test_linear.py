from __future__ import annotations

import pytest
import torch

from torml.linear_model import LinearRegression, LogisticRegression


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


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(40, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(40, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(40), torch.ones(40)]).long()
    return X, y


class TestLogisticRegression:
    def test_fit_predict(self, blobs):
        X, y = blobs
        clf = LogisticRegression().fit(X, y)
        assert clf.classes_.tolist() == [0, 1]
        assert tuple(clf.coef_.shape) == (2,)
        assert float((clf.predict(X) == y).float().mean()) > 0.9

    def test_proba_and_decision(self, blobs):
        X, y = blobs
        clf = LogisticRegression().fit(X, y)
        proba = clf.predict_proba(X[:5])
        assert tuple(proba.shape) == (5, 2)
        torch.testing.assert_close(
            proba.sum(dim=1), torch.ones(5), rtol=1e-5, atol=1e-5
        )
        scores = clf.decision_function(X)
        pred = torch.where(scores >= 0, clf.classes_[1], clf.classes_[0])
        assert torch.equal(pred, clf.predict(X))

    def test_score(self, blobs):
        X, y = blobs
        assert float(LogisticRegression().fit(X, y).score(X, y)) > 0.9

    def test_not_fitted(self):
        from torml.utils import NotFittedError

        with pytest.raises((NotFittedError, ValueError)):
            LogisticRegression().predict(torch.randn(2, 2))

    def test_multiclass_raises(self, blobs):
        X, _ = blobs
        with pytest.raises(ValueError, match="binary"):
            LogisticRegression().fit(X, torch.arange(80) % 3)

    def test_invalid_hyperparams(self, blobs):
        X, y = blobs
        with pytest.raises(ValueError, match="penalty"):
            LogisticRegression(penalty="l1").fit(X, y)
        with pytest.raises(ValueError, match="C must be"):
            LogisticRegression(C=0.0).fit(X, y)

    def test_get_params_clone(self, blobs):
        from torml.base import clone

        X, y = blobs
        clf = LogisticRegression(C=2.0).fit(X, y)
        assert clf.get_params()["C"] == 2.0
        assert isinstance(clone(clf), LogisticRegression)
