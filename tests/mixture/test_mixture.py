"""Tests for torml.mixture."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone, is_clusterer
from torml.mixture import GaussianMixture
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(40, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(40, 2) + torch.tensor([3.0, 0.0])
    return torch.cat([x0, x1])


class TestGMM:
    def test_two_components(self, blobs):
        gm = GaussianMixture(n_components=2, random_state=0).fit(blobs)
        assert is_clusterer(gm)
        assert tuple(gm.means_.shape) == (2, 2)
        assert tuple(gm.covariances_.shape) == (2, 2, 2)
        assert abs(float(gm.weights_.sum()) - 1.0) < 1e-5
        labels = gm.predict(blobs)
        assert len(torch.unique(labels)) == 2

    def test_proba_rows_sum(self, blobs):
        proba = (
            GaussianMixture(n_components=2, random_state=0)
            .fit(blobs)
            .predict_proba(blobs[:5])
        )
        assert tuple(proba.shape) == (5, 2)
        torch.testing.assert_close(
            proba.sum(dim=1), torch.ones(5), rtol=1e-4, atol=1e-4
        )

    def test_score_finite(self, blobs):
        assert torch.isfinite(
            torch.tensor(
                GaussianMixture(n_components=2, random_state=0).fit(blobs).score(blobs)
            )
        )

    def test_invalid(self, blobs):
        with pytest.raises(ValueError, match="n_components"):
            GaussianMixture(n_components=0).fit(blobs)
        with pytest.raises(ValueError, match="n_samples"):
            GaussianMixture(n_components=10).fit(blobs[:3])

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            GaussianMixture().predict(torch.randn(2, 2))

    def test_clone(self, blobs):
        assert isinstance(
            clone(GaussianMixture(n_components=2).fit(blobs)), GaussianMixture
        )
