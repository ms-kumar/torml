"""Tests for torml.covariance."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone
from torml.covariance import EmpiricalCovariance
from torml.utils import NotFittedError


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(50, 3)


class TestEmpiricalCovariance:
    def test_shapes(self, data):
        ec = EmpiricalCovariance().fit(data)
        assert tuple(ec.covariance_.shape) == (3, 3)
        assert tuple(ec.precision_.shape) == (3, 3)
        assert tuple(ec.location_.shape) == (3,)

    def test_matches_torch(self, data):
        ec = EmpiricalCovariance().fit(data)
        expected = torch.cov(data.T, correction=0)
        torch.testing.assert_close(ec.covariance_, expected, rtol=1e-4, atol=1e-4)

    def test_mahalanobis_nonnegative(self, data):
        dist = EmpiricalCovariance().fit(data).mahalanobis(data)
        assert tuple(dist.shape) == (50,)
        assert bool((dist >= 0).all())

    def test_score_finite(self, data):
        assert torch.isfinite(torch.tensor(EmpiricalCovariance().fit(data).score(data)))

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            EmpiricalCovariance().mahalanobis(torch.randn(3, 3))

    def test_clone(self, data):
        assert isinstance(clone(EmpiricalCovariance().fit(data)), EmpiricalCovariance)
