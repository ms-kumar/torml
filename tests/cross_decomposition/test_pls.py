"""Tests for torml.cross_decomposition."""

from __future__ import annotations

import pytest
import torch

from torml.cross_decomposition import PLSRegression
from torml.utils import NotFittedError


@pytest.fixture
def data():
    torch.manual_seed(0)
    X = torch.randn(40, 5)
    y = X[:, 0] * 2 - X[:, 1] + 0.1 * torch.randn(40)
    return X, y


class TestPLS:
    def test_fit_predict(self, data):
        X, y = data
        pls = PLSRegression(n_components=2).fit(X, y)
        assert tuple(pls.coef_.shape) == (5,)
        assert float(pls.score(X, y)) > 0.9

    def test_transform_shape(self, data):
        X, y = data
        assert tuple(PLSRegression(n_components=2).fit(X, y).transform(X).shape) == (
            40,
            2,
        )

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            PLSRegression().predict(torch.randn(3, 5))

    def test_invalid_components(self, data):
        X, y = data
        with pytest.raises(ValueError, match="n_components"):
            PLSRegression(n_components=0).fit(X, y)
