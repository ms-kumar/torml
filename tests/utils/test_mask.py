"""Tests for torml.utils mask utilities."""

from __future__ import annotations

import pytest
import torch

from torml.utils import indices_to_mask, safe_mask


@pytest.fixture
def X():
    torch.manual_seed(0)
    return torch.randn(6, 3)


class TestSafeMask:
    def test_passthrough(self, X):
        """Test a valid mask passes validation."""
        mask = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = safe_mask(X, mask)
        assert torch.equal(out.to(torch.float32), mask.to(torch.float32))

    def test_length_mismatch_raises(self, X):
        """Test mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="not compatible"):
            safe_mask(X, torch.ones(4))

    def test_zero_reserved_raises(self, X):
        """Test that 0 values are rejected."""
        with pytest.raises(ValueError, match="reserved"):
            safe_mask(X, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


class TestIndicesToMask:
    def test_basic(self):
        """Test index conversion."""
        mask = indices_to_mask(torch.tensor([0, 2]), 4)
        assert mask.tolist() == [1, 0, 1, 0]
        assert mask.dtype == torch.uint8

    def test_out_of_bounds_raises(self):
        """Test out-of-range indices raise ValueError."""
        with pytest.raises(ValueError, match="out-of-bounds"):
            indices_to_mask(torch.tensor([5]), 4)

    def test_bad_n_samples_raises(self):
        """Test invalid n_samples raises."""
        with pytest.raises(ValueError, match="n_samples"):
            indices_to_mask(torch.tensor([0]), -1)
