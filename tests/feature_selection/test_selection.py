"""Tests for torml.feature_selection."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone
from torml.feature_selection import SelectKBest, f_classif


@pytest.fixture
def data():
    torch.manual_seed(0)
    y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    signal = y.float().unsqueeze(1) * 3.0 + 0.5 * torch.randn(60, 2)
    noise = torch.randn(60, 3)
    return torch.cat([signal, noise], dim=1), y


class TestFClassif:
    def test_shapes(self, data):
        X, y = data
        scores, pvalues = f_classif(X, y)
        assert tuple(scores.shape) == (5,)
        assert tuple(pvalues.shape) == (5,)
        assert bool(((pvalues >= 0) & (pvalues <= 1)).all())

    def test_signal_scores_higher(self, data):
        X, y = data
        scores, _ = f_classif(X, y)
        assert float(scores[:2].min()) > float(scores[2:].max())


class TestSelectKBest:
    def test_select_two(self, data):
        X, y = data
        Xt = SelectKBest(k=2).fit_transform(X, y)
        assert tuple(Xt.shape) == (60, 2)

    def test_support(self, data):
        X, y = data
        sel = SelectKBest(k=2).fit(X, y)
        assert int(sel.get_support().sum()) == 2
        assert tuple(sel.get_support(indices=True).shape) == (2,)

    def test_all(self, data):
        X, y = data
        assert tuple(SelectKBest(k="all").fit_transform(X, y).shape) == (60, 5)

    def test_invalid_k(self, data):
        X, y = data
        with pytest.raises(ValueError, match="k="):
            SelectKBest(k=0).fit(X, y)
        with pytest.raises(ValueError, match="k="):
            SelectKBest(k=10).fit(X, y)

    def test_clone(self, data):
        X, y = data
        assert isinstance(clone(SelectKBest(k=2).fit(X, y)), SelectKBest)
