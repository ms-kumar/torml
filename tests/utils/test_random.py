"""Test random utilities."""

from __future__ import annotations

import pytest
import torch

from torml.utils._random import RandomState, check_random_state


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
        assert isinstance(rng, torch.Generator)

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


class TestRandomState:
    """Tests for RandomState."""

    def test_int_seed_reproducible(self, seed):
        """Test that int seeds give reproducible draws."""
        first = torch.randn(5, generator=RandomState(seed)())
        second = torch.randn(5, generator=RandomState(seed)())
        assert torch.equal(first, second)

    def test_generator_passthrough(self, generator):
        """Test that a generator is stored unchanged."""
        state = RandomState(generator)
        assert state() is generator
        assert state.seed is None

    def test_invalid_seed_raises(self, manual_seed):
        """Test that bad seeds raise TypeError."""
        with pytest.raises(TypeError):
            RandomState(manual_seed)
        with pytest.raises(TypeError):
            RandomState(True)

    def test_reset(self, seed):
        """Test reset resamples deterministically."""
        state = RandomState(seed)
        state.reset(seed + 1)
        assert state.seed == seed + 1
        expected = torch.randn(5, generator=RandomState(seed + 1)())
        assert torch.equal(torch.randn(5, generator=state()), expected)
        with pytest.raises(TypeError):
            state.reset("bad")

    def test_pickle_round_trip(self, seed):
        """Test pickling preserves the seed."""
        import copy
        import pickle

        state = RandomState(seed)
        assert pickle.loads(pickle.dumps(state)).seed == seed
        assert copy.copy(state).seed == seed
        assert copy.deepcopy(state).seed == seed

    def test_check_seed(self, seed, generator):
        """Test check_seed validation."""
        state = RandomState(seed)
        assert state.check_seed(seed) == seed
        assert state.check_seed(None) is None
        assert state.check_seed(generator) is generator
        with pytest.raises(TypeError):
            state.check_seed("bad")
