"""Tests for torml.multiclass."""

from __future__ import annotations

import pytest
import torch

from torml.base import is_classifier
from torml.multiclass import OneVsRestClassifier
from torml.neighbors import KNeighborsClassifier
from torml.tree import DecisionTreeClassifier
from torml.utils import NotFittedError


@pytest.fixture
def three_class():
    torch.manual_seed(0)
    x0 = torch.randn(20, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(20, 2) + torch.tensor([3.0, 0.0])
    x2 = torch.randn(20, 2) + torch.tensor([0.0, 3.0])
    X = torch.cat([x0, x1, x2])
    y = torch.cat([torch.zeros(20), torch.ones(20), torch.full((20,), 2)]).long()
    return X, y


class TestOvR:
    def test_three_classes(self, three_class):
        X, y = three_class
        clf = OneVsRestClassifier(DecisionTreeClassifier()).fit(X, y)
        assert is_classifier(clf)
        assert len(clf.estimators_) == 3
        assert float((clf.predict(X) == y).float().mean()) > 0.9

    def test_single_label_raises(self, three_class):
        X, y = three_class
        with pytest.raises(ValueError, match="at least 2"):
            OneVsRestClassifier(DecisionTreeClassifier()).fit(X, torch.zeros(60))

    def test_bad_estimator_raises(self, three_class):
        X, y = three_class
        with pytest.raises(TypeError, match="BaseEstimator"):
            OneVsRestClassifier("nope").fit(X, y)

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            OneVsRestClassifier(DecisionTreeClassifier()).predict(torch.randn(2, 2))

    def test_proba_member_scores(self, three_class):
        X, y = three_class
        clf = OneVsRestClassifier(KNeighborsClassifier(3)).fit(X, y)
        assert tuple(clf.decision_function(X).shape) == (60, 3)
