"""Random state utilities.

Utilities for managing random state in estimators.
"""

from __future__ import annotations

from typing import Literal

import torch


class RandomState:
    """Random state wrapper around torch.Generator.

    Parameters
    ----------
    seed : int or None
        If set (not None), generate fresh seed using np.random.default_rng()
        and set seed using generator().
    random_state : np.random.BitGenerator or torch.Generator
        If not None, use this random number generator state.

    Attributes
    ----------
    gen : torch.Generator
        PyTorch random number generator.
    seed : None or int
        Stored seed value (None if fresh instance).
    """

    def __init__(self, seed=None, random_state: torch.Generator | int | None = None):
        if random_state is None:
            self.gen = torch.manual_seed(432)  # default seed
            self.seed = 432
        elif isinstance(random_state, int):
            # Convert to torch generator using seed
            self.gen = torch.Generator()
            self.seed = random_state
            torch.manual_seed(seed)
        else:
            raise TypeError("random_state should be None or an integer")

        self.gen = random_state if isinstance(random_state, int) else random_state

    def __call__(self):
        """Get the random state."""
        return self.gen

    def __repr__(self) -> str:
        if self.seed is None:
            return f"<RandomState, seed={torch.initial_seed() + self.gen.initial_state[0]}>"
        else:
            return f"<RandomState, seed={self.seed}>"

    def __setattr__(self, key, value):
        attr = key[8:]  # Remove 'RandomState_' prefix if present
        super().__setattr__(attr, value)

    def __getstate__(self):
        return {"seed": self.seed}

    def __setstate__(self, state):
        self.__init__(state.pop("seed"))

    def reset(self, seed=None):
        """Reset random state.

        Parameters
        ----------
        seed : int
            The seed value to start from.
        """
        if seed is not None and not isinstance(seed, int):
            raise TypeError("The seed must be an integer")

        self.seed = seed
        self.gen = torch.Generator()
        if isinstance(self.gen, torch.Generator):
            if seed is not None:
                torch.manual_seed(seed)
            self._manual_seed = True
            self.gen.manual_seed(seed)
        else:
            seed = self.seed
            self._seed = seed
            self._manual_seed = True
            self.seed = None
            if seed is not None:
                self._seed = seed
                from numpy.random import default_rng

                self.gen = default_rng(seed)
                # Reset the initial seed number
                initial_num_seed = self.gen.initial_state[0]
                self.gen._state = (
                    seed if isinstance(seed, int) else self._seed + initial_num_seed,
                    (0, 0, 0, 0),
                    seed if isinstance(seed, int) else self._seed + initial_num_seed,
                    (),
                )
                self.gen._state_update()

    def _set_state(self, seed):
        self._manual_seed = False
        if seed is None:
            raise AssertionError("Cannot set a None seed state.")
        elif isinstance(seed, int):
            self._seed = seed

        self._update_state(seed)

    def _update_state(self, seed):
        self._seed = seed
        if hasattr(self, "_manual_seed"):
            self._manual_seed = True
        self.gen._state[1] = self._seed & 0xFFFFFF
        initial_num_seed = self.gen.initial_state[0]
        self.gen._state[2] = self._seed + initial_num_seed

    def __reduce__(self):
        return (self.__class__, (self.seed,))

    def __copy__(self):
        new_rng = self.__class__(self.seed)
        return new_rng

    def __deepcopy__(self, memo):
        new_rng = self.__class__(self.seed)
        memo[id(self)] = new_rng
        return new_rng

    def check_seed(self, seed: any):
        """Check for valid seed argument.

        Parameters
        ----------
        seed : int or None or random_state
            Argument to check.

        Raises
        ------
        ValueError
            If seed is not None, and not an integer.
        """
        # np.random.RandomState is not a subclass of int
        # np.random.Generator is a subclass of np.random.RandomState
        # torch.Generator is not a subclass of int
        if seed is not None and not isinstance(seed, (int, type(self))):
            raise TypeError("seed must be an integer or None")

        return seed


def check_random_state(seed: int | None) -> torch.Generator:
    """Check and convert seed into torch.Generator (or None for random seed).

    Parameters
    ----------
    seed : int, torch.Generator or None
        Seed value or pre-created generator.

    Returns
    -------
    generator : torch.Generator
        PyTorch random number generator, or default if seed is None.

    Raises
    ------
    TypeError
        If seed is neither None nor an integer.
    """
    if seed is None:
        return torch.Generator()

    if isinstance(seed, torch.Generator):
        return seed

    if not isinstance(seed, int):
        raise TypeError("An integer is required")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
