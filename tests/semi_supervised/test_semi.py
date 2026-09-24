"""Tests for torml.semi_supervised."""

from __future__ import annotations

import pytest
import torch

from torml.base import is_classifier
from torml.semi_supervised import LabelPropagation
from torml.utils import NotFittedError


@pytest.fixture
def partial_blobs():
    torch.manual_seed(0)
    x0 = torch.randn(20, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(20, 2) + torch.tensor([3.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    y[5:15] = -1
    y[25:35] = -1
    return X, y


class TestLabelPropagation:
    def test_recovers_labels(self, partial_blobs):
        X, y = partial_blobs
        lp = LabelPropagation().fit(X, y)
        assert is_classifier(lp)
        assert tuple(lp.transduction_.shape) == (40,)
        assert (
            float(
                (lp.predict(X) == torch.cat([torch.zeros(20), torch.ones(20)]))
                .float()
                .mean()
            )
            > 0.9
        )

    def test_proba(self, partial_blobs):
        X, y = partial_blobs
        proba = LabelPropagation().fit(X, y).predict_proba(X)
        assert tuple(proba.shape) == (40, 2)

    def test_no_labels_raises(self, partial_blobs):
        X, _ = partial_blobs
        with pytest.raises(ValueError, match="At least one labeled"):
            LabelPropagation().fit(X, torch.full((40,), -1))

    def test_not_fitted(self, partial_blobs):
        X, _ = partial_blobs
        with pytest.raises((NotFittedError, ValueError)):
            LabelPropagation().predict(X)

    def test_transductive_mismatch(self, partial_blobs):
        X, y = partial_blobs
        lp = LabelPropagation().fit(X, y)
        with pytest.raises(ValueError, match="transductive"):
            lp.predict(torch.randn(5, 2))
