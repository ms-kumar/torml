"""Tests for torml.random_projection."""

from __future__ import annotations

import pytest
import torch

from torml.random_projection import (
    GaussianRandomProjection,
    johnson_lindenstrauss_min_dim,
)


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(30, 10)


class TestGRP:
    def test_fixed_dim(self, data):
        Xt = GaussianRandomProjection(n_components=4, random_state=0).fit_transform(
            data
        )
        assert tuple(Xt.shape) == (30, 4)

    def test_deterministic(self, data):
        a = GaussianRandomProjection(n_components=4, random_state=1).fit_transform(data)
        b = GaussianRandomProjection(n_components=4, random_state=1).fit_transform(data)
        assert torch.equal(a, b)

    def test_inverse_shape(self, data):
        g = GaussianRandomProjection(n_components=4, random_state=0).fit(data)
        assert tuple(g.inverse_transform(g.transform(data)).shape) == (30, 10)

    def test_jl_bound(self):
        assert johnson_lindenstrauss_min_dim(100) > johnson_lindenstrauss_min_dim(10)
        with pytest.raises(ValueError, match="eps"):
            johnson_lindenstrauss_min_dim(100, eps=1.5)

    def test_invalid(self, data):
        with pytest.raises(ValueError, match="n_components"):
            GaussianRandomProjection(n_components=0).fit(data)
