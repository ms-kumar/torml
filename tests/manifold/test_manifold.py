"""Tests for torml.manifold."""

from __future__ import annotations

import pytest
import torch

from torml.manifold import MDS
from torml.utils import NotFittedError


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(20, 4)


class TestMDS:
    def test_embedding_shape(self, data):
        emb = MDS(n_components=2).fit_transform(data)
        assert tuple(emb.shape) == (20, 2)

    def test_preserves_distances(self, data):
        mds = MDS(n_components=4).fit(data)
        d_orig = torch.cdist(data, data)
        d_emb = torch.cdist(mds.embedding_, mds.embedding_)
        assert float(((d_orig - d_emb) ** 2).mean()) < 0.5

    def test_precomputed(self, data):
        dist = torch.cdist(data, data)
        emb = MDS(n_components=2, dissimilarity="precomputed").fit_transform(dist)
        assert tuple(emb.shape) == (20, 2)

    def test_score(self, data):
        assert MDS(n_components=2).fit(data).score(data) <= 0

    def test_invalid(self, data):
        with pytest.raises(ValueError, match="n_components"):
            MDS(n_components=0).fit(data)
        with pytest.raises(ValueError, match="dissimilarity"):
            MDS(dissimilarity="bad").fit(data)

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            MDS().score(torch.randn(5, 3))
