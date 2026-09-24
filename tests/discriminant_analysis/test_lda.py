"""Tests for torml.discriminant_analysis."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone, is_classifier
from torml.discriminant_analysis import LinearDiscriminantAnalysis
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(30, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(30, 2) + torch.tensor([3.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    return X, y


class TestLDA:
    def test_fit_predict(self, blobs):
        X, y = blobs
        clf = LinearDiscriminantAnalysis().fit(X, y)
        assert is_classifier(clf)
        assert float((clf.predict(X) == y).float().mean()) > 0.95

    def test_proba(self, blobs):
        X, y = blobs
        proba = LinearDiscriminantAnalysis().fit(X, y).predict_proba(X[:5])
        assert tuple(proba.shape) == (5, 2)
        torch.testing.assert_close(
            proba.sum(dim=1), torch.ones(5), rtol=1e-4, atol=1e-4
        )

    def test_transform_shape(self, blobs):
        X, y = blobs
        assert tuple(LinearDiscriminantAnalysis().fit(X, y).transform(X).shape) == (
            60,
            1,
        )

    def test_single_class_raises(self, blobs):
        X, _ = blobs
        with pytest.raises(ValueError, match="at least 2"):
            LinearDiscriminantAnalysis().fit(X, torch.zeros(60))

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            LinearDiscriminantAnalysis().predict(torch.randn(2, 2))

    def test_clone(self, blobs):
        X, y = blobs
        assert isinstance(
            clone(LinearDiscriminantAnalysis().fit(X, y)), LinearDiscriminantAnalysis
        )
