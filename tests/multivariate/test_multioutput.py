"""Tests for torml.multivariate."""

from __future__ import annotations

import pytest
import torch

from torml.linear_model import LinearRegression
from torml.multivariate import MultiOutputClassifier, MultiOutputRegressor
from torml.tree import DecisionTreeClassifier
from torml.utils import NotFittedError


@pytest.fixture
def regression_data():
    torch.manual_seed(0)
    X = torch.randn(40, 3)
    Y = torch.stack([X[:, 0] * 2, X[:, 1] * -1], dim=1)
    return X, Y


class TestMultiOutputRegressor:
    def test_two_outputs(self, regression_data):
        X, Y = regression_data
        pred = MultiOutputRegressor(LinearRegression()).fit(X, Y).predict(X)
        assert tuple(pred.shape) == (40, 2)
        torch.testing.assert_close(pred, Y, rtol=1e-3, atol=1e-3)

    def test_single_output(self, regression_data):
        X, Y = regression_data
        pred = MultiOutputRegressor(LinearRegression()).fit(X, Y[:, 0]).predict(X)
        assert tuple(pred.shape) == (40,)

    def test_bad_estimator(self, regression_data):
        X, Y = regression_data
        with pytest.raises(TypeError, match="BaseEstimator"):
            MultiOutputRegressor("nope").fit(X, Y)

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            MultiOutputRegressor(LinearRegression()).predict(torch.randn(2, 3))


class TestMultiOutputClassifier:
    def test_two_outputs(self):
        torch.manual_seed(0)
        X = torch.randn(40, 2)
        Y = torch.stack([(X[:, 0] > 0).long(), (X[:, 1] > 0).long()], dim=1)
        pred = MultiOutputClassifier(DecisionTreeClassifier()).fit(X, Y).predict(X)
        assert tuple(pred.shape) == (40, 2)
        assert float((pred == Y).float().mean()) > 0.9


class TestNestedParams:
    def test_estimator_prefix_round_trip(self):
        """Test estimator__param get/set round-trip."""
        reg = MultiOutputRegressor(LinearRegression())
        assert reg.get_params()["estimator__fit_intercept"] is True
        reg.set_params(estimator__fit_intercept=False)
        assert reg.get_params()["estimator__fit_intercept"] is False
        clf = MultiOutputClassifier(DecisionTreeClassifier())
        assert clf.get_params()["estimator__criterion"] == "gini"
        clf.set_params(estimator__max_depth=2)
        assert clf.get_params()["estimator__max_depth"] == 2
