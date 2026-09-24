"""Tests for torml.tree."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone, is_classifier, is_regressor
from torml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(30, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(30, 2) + torch.tensor([3.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    return X, y


@pytest.fixture
def regression_data():
    torch.manual_seed(1)
    X = torch.linspace(-3, 3, 60).unsqueeze(1)
    y = X.squeeze(1) ** 2
    return X, y


class TestDecisionTreeClassifier:
    def test_fit_predict(self, blobs):
        X, y = blobs
        clf = DecisionTreeClassifier().fit(X, y)
        assert is_classifier(clf)
        assert clf.classes_.tolist() == [0, 1]
        assert float((clf.predict(X) == y).float().mean()) == 1.0

    def test_proba(self, blobs):
        X, y = blobs
        proba = DecisionTreeClassifier().fit(X, y).predict_proba(X[:5])
        assert tuple(proba.shape) == (5, 2)
        torch.testing.assert_close(
            proba.sum(dim=1), torch.ones(5), rtol=1e-5, atol=1e-5
        )

    def test_max_depth_limits(self, blobs):
        X, y = blobs
        clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
        assert float((clf.predict(X) == y).float().mean()) > 0.9

    def test_importances_normalized(self, blobs):
        X, y = blobs
        imp = DecisionTreeClassifier().fit(X, y).feature_importances_
        assert abs(float(imp.sum()) - 1.0) < 1e-5

    def test_entropy(self, blobs):
        X, y = blobs
        clf = DecisionTreeClassifier(criterion="entropy").fit(X, y)
        assert float((clf.predict(X) == y).float().mean()) == 1.0

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            DecisionTreeClassifier().predict(torch.randn(2, 2))

    def test_invalid_criterion(self, blobs):
        X, y = blobs
        with pytest.raises(ValueError, match="criterion"):
            DecisionTreeClassifier(criterion="bad").fit(X, y)

    def test_clone(self, blobs):
        X, y = blobs
        clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
        assert isinstance(clone(clf), DecisionTreeClassifier)


class TestDecisionTreeRegressor:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        reg = DecisionTreeRegressor().fit(X, y)
        assert is_regressor(reg)
        pred = reg.predict(X)
        assert tuple(pred.shape) == (60,)
        assert float(reg.score(X, y)) > 0.99

    def test_constant_leaf(self):
        X = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        y = torch.tensor([2.0, 2.0, 5.0, 5.0])
        pred = DecisionTreeRegressor().fit(X, y).predict(X)
        torch.testing.assert_close(pred, y, rtol=1e-5, atol=1e-5)

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            DecisionTreeRegressor().predict(torch.randn(2, 1))
