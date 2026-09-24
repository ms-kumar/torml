"""Tests for torml.gaussian_process."""

from __future__ import annotations

import pytest
import torch

from torml.base import clone, is_regressor
from torml.gaussian_process import GaussianProcessRegressor
from torml.utils import NotFittedError


@pytest.fixture
def data():
    torch.manual_seed(0)
    X = torch.linspace(-2, 2, 15).unsqueeze(1)
    y = torch.sin(X.squeeze(1))
    return X, y


class TestGPR:
    def test_interpolates(self, data):
        X, y = data
        gpr = GaussianProcessRegressor(alpha=1e-8).fit(X, y)
        assert is_regressor(gpr)
        torch.testing.assert_close(gpr.predict(X), y, rtol=1e-3, atol=1e-3)

    def test_std(self, data):
        X, y = data
        mean, std = GaussianProcessRegressor().fit(X, y).predict(X, return_std=True)
        assert tuple(std.shape) == (15,)
        assert bool((std >= 0).all())

    def test_lml_finite(self, data):
        X, y = data
        assert torch.isfinite(
            torch.tensor(GaussianProcessRegressor().fit(X, y).log_marginal_likelihood())
        )

    def test_invalid(self, data):
        X, y = data
        with pytest.raises(ValueError, match="length_scale"):
            GaussianProcessRegressor(length_scale=0.0).fit(X, y)
        with pytest.raises(ValueError, match="alpha"):
            GaussianProcessRegressor(alpha=-1.0).fit(X, y)

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            GaussianProcessRegressor().predict(torch.randn(2, 1))

    def test_clone(self, data):
        X, y = data
        assert isinstance(
            clone(GaussianProcessRegressor().fit(X, y)), GaussianProcessRegressor
        )
