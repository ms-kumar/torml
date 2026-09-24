"""Tests for torml.svm."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone, is_classifier, is_regressor
from torml.svm import LinearSVC, LinearSVR
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(40, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(40, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(40), torch.ones(40)]).long()
    return X, y


@pytest.fixture
def regression_data():
    torch.manual_seed(1)
    X = torch.randn(60, 2)
    y = 2 * X[:, 0] - X[:, 1]
    return X, y


class TestLinearSVC:
    def test_fit_predict(self, blobs):
        X, y = blobs
        clf = LinearSVC(random_state=0).fit(X, y)
        assert is_classifier(clf)
        assert clf.classes_.tolist() == [0, 1]
        assert float((clf.predict(X) == y).float().mean()) > 0.9

    def test_decision_function_sign(self, blobs):
        X, y = blobs
        clf = LinearSVC(random_state=0).fit(X, y)
        scores = clf.decision_function(X)
        assert tuple(scores.shape) == (80,)
        pred = torch.where(scores >= 0, clf.classes_[1], clf.classes_[0])
        assert torch.equal(pred, clf.predict(X))

    def test_multiclass_raises(self, blobs):
        X, _ = blobs
        y = torch.arange(80) % 3
        with pytest.raises(ValueError, match="binary"):
            LinearSVC().fit(X, y)

    def test_invalid_C(self, blobs):
        X, y = blobs
        with pytest.raises(ValueError, match="C must be"):
            LinearSVC(C=0.0).fit(X, y)

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            LinearSVC().predict(torch.randn(2, 2))

    def test_clone(self, blobs):
        X, y = blobs
        clf = LinearSVC(C=2.0).fit(X, y)
        assert isinstance(clone(clf), LinearSVC)


class TestLinearSVR:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        reg = LinearSVR(random_state=0).fit(X, y)
        assert is_regressor(reg)
        assert tuple(reg.predict(X).shape) == (60,)
        assert float(reg.score(X, y)) > 0.8

    def test_invalid_epsilon(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="epsilon"):
            LinearSVR(epsilon=-0.1).fit(X, y)

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            LinearSVR().predict(torch.randn(2, 2))
