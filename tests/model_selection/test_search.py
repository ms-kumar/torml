"""Tests for torml.model_selection.GridSearchCV."""

from __future__ import annotations

import pytest
import torch

from torml.linear_model import LinearRegression
from torml.model_selection import GridSearchCV
from torml.neighbors import KNeighborsClassifier
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(30, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(30, 2) + torch.tensor([3.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    return X, y


class TestGridSearchCV:
    def test_finds_k(self, blobs):
        X, y = blobs
        gs = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [1, 5]}, cv=3).fit(
            X, y
        )
        assert gs.best_params_["n_neighbors"] in (1, 5)
        assert tuple(gs.predict(X).shape) == (60,)
        assert float(gs.score(X, y)) > 0.9

    def test_cv_results_ordered(self, blobs):
        X, y = blobs
        gs = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [1, 3, 7]}, cv=2).fit(
            X, y
        )
        scores = gs.cv_results_["mean_test_score"]
        assert scores == sorted(scores, reverse=True)
        assert len(gs.cv_results_["params"]) == 3

    def test_no_refit(self, blobs):
        X, y = blobs
        gs = GridSearchCV(
            KNeighborsClassifier(), {"n_neighbors": [3]}, cv=2, refit=False
        ).fit(X, y)
        with pytest.raises(Exception, match="not fitted|NotFitted"):
            gs.predict(X)

    def test_regression(self):
        torch.manual_seed(1)
        X = torch.randn(40, 2)
        y = 2 * X[:, 0] - X[:, 1]
        gs = GridSearchCV(
            LinearRegression(), {"fit_intercept": [True, False]}, cv=2
        ).fit(X, y)
        assert gs.best_params_["fit_intercept"] is True

    def test_invalid_grid(self, blobs):
        X, y = blobs
        with pytest.raises(ValueError, match="non-empty dict"):
            GridSearchCV(KNeighborsClassifier(), {}).fit(X, y)
        with pytest.raises(TypeError, match="dict or list"):
            GridSearchCV(KNeighborsClassifier(), "bad").fit(X, y)

    def test_not_fitted(self, blobs):
        X, _ = blobs
        with pytest.raises((NotFittedError, ValueError)):
            GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [3]}).predict(X)
