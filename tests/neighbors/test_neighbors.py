"""Tests for torml.neighbors."""

from __future__ import annotations

import pytest
import torch

from torml.base import is_classifier, is_regressor
from torml.neighbors import KNeighborsClassifier, KNeighborsRegressor
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(20, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(20, 2) + torch.tensor([3.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    return X, y


@pytest.fixture
def regression_data():
    torch.manual_seed(1)
    X = torch.linspace(-3, 3, 30).unsqueeze(1)
    y = X.squeeze(1) ** 2
    return X, y


class TestKNeighborsClassifier:
    def test_fit_predict_separable(self, blobs):
        X, y = blobs
        clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        assert is_classifier(clf)
        assert not is_regressor(clf)
        pred = clf.predict(X)
        assert tuple(pred.shape) == (40,)
        assert float((pred == y).float().mean()) > 0.95

    def test_predict_proba_sums_to_one(self, blobs):
        X, y = blobs
        proba = KNeighborsClassifier(n_neighbors=3).fit(X, y).predict_proba(X[:5])
        assert tuple(proba.shape) == (5, 2)
        torch.testing.assert_close(
            proba.sum(dim=1), torch.ones(5), rtol=1e-5, atol=1e-5
        )

    def test_weights_distance(self, blobs):
        X, y = blobs
        pred = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X, y)
        assert float((pred.predict(X) == y).float().mean()) > 0.95

    def test_manhattan(self, blobs):
        X, y = blobs
        pred = KNeighborsClassifier(n_neighbors=3, p=1).fit(X, y).predict(X)
        assert float((pred == y).float().mean()) > 0.9

    def test_kneighbors_shapes(self, blobs):
        X, y = blobs
        clf = KNeighborsClassifier(n_neighbors=4).fit(X, y)
        dist, ind = clf.kneighbors(X[:6])
        assert tuple(dist.shape) == (6, 4)
        assert tuple(ind.shape) == (6, 4)
        assert bool((dist >= 0).all())

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            KNeighborsClassifier().predict(torch.randn(2, 2))

    def test_feature_mismatch_raises(self, blobs):
        X, y = blobs
        clf = KNeighborsClassifier().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            clf.predict(torch.randn(3, 5))

    def test_invalid_hyperparams(self, blobs):
        X, y = blobs
        with pytest.raises(ValueError, match="n_neighbors"):
            KNeighborsClassifier(n_neighbors=0).fit(X, y)
        with pytest.raises(ValueError, match="weights"):
            KNeighborsClassifier(weights="bad").fit(X, y)
        with pytest.raises(ValueError, match="n_neighbors"):
            KNeighborsClassifier(n_neighbors=100).fit(X, y)

    def test_get_params_round_trip(self):
        clf = KNeighborsClassifier(n_neighbors=3, p=1)
        assert clf.get_params()["n_neighbors"] == 3
        assert clf.set_params(n_neighbors=7).n_neighbors == 7


class TestKNeighborsRegressor:
    def test_fit_predict_on_train_points(self, regression_data):
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=1).fit(X, y)
        assert is_regressor(reg)
        torch.testing.assert_close(reg.predict(X), y, rtol=1e-4, atol=1e-4)

    def test_uniform_smoothing(self, regression_data):
        X, y = regression_data
        pred = KNeighborsRegressor(n_neighbors=5).fit(X, y).predict(X)
        assert tuple(pred.shape) == (30,)
        assert torch.isfinite(pred).all()

    def test_distance_weights(self, regression_data):
        X, y = regression_data
        pred = (
            KNeighborsRegressor(n_neighbors=3, weights="distance").fit(X, y).predict(X)
        )
        torch.testing.assert_close(pred, y, rtol=1e-2, atol=1e-1)

    def test_score_high(self, regression_data):
        X, y = regression_data
        score = KNeighborsRegressor(n_neighbors=3).fit(X, y).score(X, y)
        assert float(score) > 0.9

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            KNeighborsRegressor().predict(torch.randn(2, 1))

    def test_invalid_hyperparams(self, regression_data):
        X, y = regression_data
        with pytest.raises(TypeError, match="n_neighbors"):
            KNeighborsRegressor(n_neighbors=2.5).fit(X, y)
        with pytest.raises(ValueError, match="p must be"):
            KNeighborsRegressor(p=0.5).fit(X, y)
