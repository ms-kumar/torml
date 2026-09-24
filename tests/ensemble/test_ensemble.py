"""Tests for torml.ensemble."""

from __future__ import annotations

import pytest
import torch

from torml.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from torml.linear_model import LinearRegression
from torml.neighbors import KNeighborsClassifier
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
    X = torch.randn(40, 2)
    y = 2 * X[:, 0] - X[:, 1]
    return X, y


class TestVoting:
    def test_hard_vote(self, blobs):
        X, y = blobs
        clf = VotingClassifier(
            [("tree", DecisionTreeClassifier()), ("knn", KNeighborsClassifier(3))]
        ).fit(X, y)
        assert float((clf.predict(X) == y).float().mean()) > 0.95

    def test_soft_vote(self, blobs):
        X, y = blobs
        clf = VotingClassifier(
            [("tree", DecisionTreeClassifier()), ("knn", KNeighborsClassifier(3))],
            voting="soft",
        ).fit(X, y)
        proba = clf.predict_proba(X[:5])
        assert tuple(proba.shape) == (5, 2)
        assert float((clf.predict(X) == y).float().mean()) > 0.95

    def test_nested_params(self, blobs):
        X, y = blobs
        clf = VotingClassifier([("tree", DecisionTreeClassifier(max_depth=1))])
        assert clf.get_params()["tree__max_depth"] == 1
        clf.set_params(tree__max_depth=2)
        assert clf.get_params()["tree__max_depth"] == 2
        clf.fit(X, y)

    def test_regressor(self, regression_data):
        X, y = regression_data
        pred = (
            VotingRegressor(
                [("lin", LinearRegression()), ("tree", DecisionTreeRegressor())]
            )
            .fit(X, y)
            .predict(X)
        )
        assert tuple(pred.shape) == (40,)

    def test_empty_raises(self, blobs):
        X, y = blobs
        with pytest.raises(TypeError, match="non-empty"):
            VotingClassifier([]).fit(X, y)


class TestBagging:
    def test_classifier(self, blobs):
        X, y = blobs
        clf = BaggingClassifier(n_estimators=5, random_state=0).fit(X, y)
        assert len(clf.estimators_) == 5
        assert float((clf.predict(X) == y).float().mean()) > 0.9

    def test_regressor(self, regression_data):
        X, y = regression_data
        reg = BaggingRegressor(n_estimators=5, random_state=0).fit(X, y)
        assert tuple(reg.predict(X).shape) == (40,)
        assert float(reg.score(X, y)) > 0.8

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            BaggingClassifier().predict(torch.randn(2, 2))


class TestForest:
    def test_classifier(self, blobs):
        X, y = blobs
        clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        assert len(clf.estimators_) == 5
        assert float((clf.predict(X) == y).float().mean()) > 0.9

    def test_regressor(self, regression_data):
        X, y = regression_data
        reg = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
        assert float(reg.score(X, y)) > 0.8

    def test_deterministic(self, blobs):
        X, y = blobs
        a = RandomForestClassifier(n_estimators=3, random_state=2).fit(X, y).predict(X)
        b = RandomForestClassifier(n_estimators=3, random_state=2).fit(X, y).predict(X)
        assert torch.equal(a, b)
