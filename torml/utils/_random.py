"""Random state utilities.

Utilities for managing random state in estimators.
"""

from __future__ import annotations

import torch


class RandomState:
    """Random state wrapper around torch.Generator.

    Parameters
    ----------
    seed : int, torch.Generator or None, default=None
        Seed value, pre-created generator, or None for an unseeded
        generator.

    Attributes
    ----------
    gen : torch.Generator
        PyTorch random number generator.
    seed : int or None
        Stored seed value (None when unseeded or wrapping a generator).
    """

    def __init__(self, seed: torch.Generator | int | None = None):
        if seed is None:
            self.gen = torch.Generator()
            self.seed = None
        elif isinstance(seed, torch.Generator):
            self.gen = seed
            self.seed = None
        elif isinstance(seed, int) and not isinstance(seed, bool):
            self.gen = torch.Generator()
            self.gen.manual_seed(seed)
            self.seed = seed
        else:
            raise TypeError(
                "seed must be an int, torch.Generator or None, "
                f"got {type(seed).__name__}."
            )

    def __call__(self):
        """Get the random state."""
        return self.gen

    def __repr__(self) -> str:
        return f"<RandomState, seed={self.seed}>"

    def __getstate__(self):
        return {"seed": self.seed}

    def __setstate__(self, state):
        self.__init__(state["seed"])

    def reset(self, seed=None):
        """Reset to a fresh generator, optionally reseeded.

        Parameters
        ----------
        seed : int or None, default=None
            New seed value.
        """
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("The seed must be an integer or None")
        self.gen = torch.Generator()
        self.seed = seed
        if seed is not None:
            self.gen.manual_seed(seed)

    def __reduce__(self):
        return (self.__class__, (self.seed,))

    def __copy__(self):
        return self.__class__(self.seed)

    def __deepcopy__(self, memo):
        new_rng = self.__class__(self.seed)
        memo[id(self)] = new_rng
        return new_rng

    def check_seed(self, seed):
        """Check for valid seed argument.

        Parameters
        ----------
        seed : int, torch.Generator, RandomState or None
            Argument to check.

        Returns
        -------
        seed : valid seed argument, unchanged.

        Raises
        ------
        TypeError
            If seed is none of the accepted types.
        """
        if seed is None or isinstance(seed, (int, torch.Generator, RandomState)):
            if isinstance(seed, bool):
                raise TypeError("seed must be an integer, Generator or None")
            return seed
        raise TypeError("seed must be an integer, Generator or None")


def check_random_state(seed: int | None, device=None) -> torch.Generator:
    """Check and convert seed into torch.Generator (or None for random seed).

    Parameters
    ----------
    seed : int, torch.Generator or None
        Seed value or pre-created generator. A passed-in generator is
        returned unchanged (it must already live on ``device``).
    device : torch.device or str or None, default=None
        Device for a newly created generator. CUDA/MPS generators are
        required to sample tensors on those devices.

    Returns
    -------
    generator : torch.Generator
        PyTorch random number generator on ``device`` (CPU default).

    Raises
    ------
    TypeError
        If seed is neither None, an integer, nor a generator.
    """
    if seed is None:
        return (
            torch.Generator(device=device) if device is not None else torch.Generator()
        )

    if isinstance(seed, torch.Generator):
        return seed

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("An integer is required")

    generator = (
        torch.Generator(device=device) if device is not None else torch.Generator()
    )
    generator.manual_seed(seed)
    return generator
