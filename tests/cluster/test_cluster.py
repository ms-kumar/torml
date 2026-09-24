"""Tests for torml.cluster."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone, is_clusterer
from torml.cluster import DBSCAN, KMeans
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(30, 2) + torch.tensor([-4.0, 0.0])
    x1 = torch.randn(30, 2) + torch.tensor([4.0, 0.0])
    return torch.cat([x0, x1])


class TestKMeans:
    def test_two_blobs(self, blobs):
        km = KMeans(n_clusters=2, random_state=0, n_init=5).fit(blobs)
        assert is_clusterer(km)
        assert tuple(km.cluster_centers_.shape) == (2, 2)
        assert tuple(km.labels_.shape) == (60,)
        assert len(torch.unique(km.labels_)) == 2
        assert km.inertia_ > 0
        assert km.predict(blobs[:4]).shape == (4,)

    def test_fit_predict_matches_labels(self, blobs):
        km = KMeans(n_clusters=2, random_state=0, n_init=2)
        assert torch.equal(km.fit_predict(blobs), km.labels_)

    def test_deterministic(self, blobs):
        a = KMeans(n_clusters=2, random_state=1, n_init=3).fit(blobs).labels_
        b = KMeans(n_clusters=2, random_state=1, n_init=3).fit(blobs).labels_
        assert torch.equal(a, b)

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            KMeans().predict(torch.randn(3, 2))

    def test_invalid_hyperparams(self, blobs):
        with pytest.raises(ValueError, match="n_clusters"):
            KMeans(n_clusters=0).fit(blobs)
        with pytest.raises(ValueError, match="n_samples"):
            KMeans(n_clusters=100).fit(blobs[:5])

    def test_get_params_clone(self, blobs):
        km = KMeans(n_clusters=3).set_params(n_clusters=2)
        assert km.get_params()["n_clusters"] == 2
        assert isinstance(clone(km), KMeans)


class TestDBSCAN:
    def test_two_blobs(self, blobs):
        db = DBSCAN(eps=1.5, min_samples=3).fit(blobs)
        assert is_clusterer(db)
        assert tuple(db.labels_.shape) == (60,)
        found = {int(v) for v in db.labels_.tolist()} - {-1}
        assert len(found) == 2

    def test_noise_label(self):
        X = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
        labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(X)
        assert int(labels[2].item()) == -1

    def test_invalid_hyperparams(self, blobs):
        with pytest.raises(ValueError, match="eps"):
            DBSCAN(eps=0.0).fit(blobs)
        with pytest.raises(TypeError, match="min_samples"):
            DBSCAN(min_samples=2.5).fit(blobs)
