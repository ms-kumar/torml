"""Tests for torml.decomposition.PCA."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone
from torml.decomposition import PCA
from torml.utils import NotFittedError


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(40, 5)


class TestPCA:
    def test_shapes(self, data):
        pca = PCA(n_components=2).fit(data)
        assert tuple(pca.components_.shape) == (2, 5)
        assert tuple(pca.transform(data).shape) == (40, 2)
        assert tuple(pca.fit_transform(data).shape) == (40, 2)

    def test_variance_sums_to_one(self, data):
        pca = PCA().fit(data)
        assert abs(float(pca.explained_variance_ratio_.sum()) - 1.0) < 1e-5

    def test_round_trip_full(self, data):
        pca = PCA().fit(data)
        torch.testing.assert_close(
            pca.inverse_transform(pca.transform(data)), data, rtol=1e-4, atol=1e-4
        )

    def test_float_ratio(self, data):
        pca = PCA(n_components=0.9).fit(data)
        assert float(pca.explained_variance_ratio_.sum()) >= 0.9 - 1e-5

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            PCA().transform(torch.randn(4, 3))

    def test_invalid_n_components(self, data):
        with pytest.raises(ValueError, match="n_components"):
            PCA(n_components=0).fit(data)
        with pytest.raises(ValueError, match="n_components"):
            PCA(n_components=100).fit(data)

    def test_clone_params(self):
        pca = PCA(n_components=2).set_params(n_components=3)
        assert pca.get_params()["n_components"] == 3
        assert isinstance(clone(pca), PCA)
