"""Tests for torml.naive_bayes.GaussianNB."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone, is_classifier
from torml.naive_bayes import GaussianNB
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(30, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(30, 2) + torch.tensor([2.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    return X, y


class TestGaussianNB:
    def test_fit_predict_separable(self, blobs):
        X, y = blobs
        clf = GaussianNB().fit(X, y)
        assert is_classifier(clf)
        assert clf.classes_.tolist() == [0, 1]
        assert tuple(clf.theta_.shape) == (2, 2)
        assert tuple(clf.var_.shape) == (2, 2)
        pred = clf.predict(X)
        assert float((pred == y).float().mean()) > 0.95

    def test_priors_and_counts(self, blobs):
        X, y = blobs
        clf = GaussianNB().fit(X, y)
        assert clf.class_count_.tolist() == [30.0, 30.0]
        torch.testing.assert_close(
            clf.class_prior_, torch.tensor([0.5, 0.5]), rtol=1e-5, atol=1e-5
        )

    def test_proba_sums_to_one(self, blobs):
        X, y = blobs
        proba = GaussianNB().fit(X, y).predict_proba(X[:6])
        assert tuple(proba.shape) == (6, 2)
        torch.testing.assert_close(
            proba.sum(dim=1), torch.ones(6), rtol=1e-4, atol=1e-4
        )
        log_proba = GaussianNB().fit(X, y).predict_log_proba(X[:6])
        torch.testing.assert_close(torch.exp(log_proba), proba, rtol=1e-4, atol=1e-4)

    def test_score_and_clone(self, blobs):
        X, y = blobs
        clf = GaussianNB().fit(X, y)
        assert float(clf.score(X, y)) > 0.95
        assert isinstance(clone(clf), GaussianNB)

    def test_string_labels(self):
        X = torch.tensor([[0.0], [1.0], [0.2], [0.9]])
        clf = GaussianNB().fit(X, ["a", "b", "a", "b"])
        assert list(clf.classes_) == ["a", "b"]
        assert clf.predict(torch.tensor([[0.1]])) == ["a"]

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            GaussianNB().predict(torch.randn(2, 2))

    def test_feature_mismatch_raises(self, blobs):
        X, y = blobs
        clf = GaussianNB().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            clf.predict(torch.randn(3, 5))

    def test_invalid_smoothing(self, blobs):
        X, y = blobs
        with pytest.raises(ValueError, match="var_smoothing"):
            GaussianNB(var_smoothing=-1.0).fit(X, y)
        with pytest.raises(TypeError, match="var_smoothing"):
            GaussianNB(var_smoothing="bad").fit(X, y)

    def test_get_params_round_trip(self):
        clf = GaussianNB(var_smoothing=1e-8)
        assert clf.get_params()["var_smoothing"] == 1e-8
        assert clf.set_params(var_smoothing=1e-7).var_smoothing == 1e-7
