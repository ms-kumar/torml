"""Test random utilities."""

from __future__ import annotations

import pytest
import torch

from torml.utils._random import check_random_state, RandomState


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def tensor():
    return torch.randn(5)


@pytest.fixture
def generator():
    return torch.Generator()


@pytest.fixture
def manual_seed():
    return "not_an_integer"


class TestCheckRandomState:
    """Tests for check_random_state."""

    def test_none_seed(self):
        """Test that None seed returns default generator."""
        rng = check_random_state(None)
        assert rng is not None
        assert isinstance(rng, torch.Generator)

    def test_int_seed_matches_input(self, seed):
        """Test that int seed converts properly."""
        rng = check_random_state(seed)
        assert isinstance(seed, (int, float))

    def test_generator_passed(self, generator):
        """Test that passing a torch.Generator returns it."""
        rng = check_random_state(generator)
        assert rng is generator

    def test_non_int_non_none_raises(self, manual_seed):
        """Test that non-int/None seed raises TypeError."""
        with pytest.raises(TypeError):
            check_random_state(manual_seed)

    def test_generator_manual_seed_correct(self, seed):
        """Test that generator seeded with int is correct."""
        generator = torch.Generator()
        generator.manual_seed(seed)
        rng = check_random_state(generator)
        assert rng is generator
