"""Splitting utilities for model selection.

Provides :class:`KFold` and :func:`train_test_split` with a PyTorch backend.
"""

from __future__ import annotations

import math

import torch

from torml.base import BaseEstimator
from torml.utils._random import check_random_state


def _num_samples(x) -> int:
    """Return number of samples in array-like."""
    if isinstance(x, torch.Tensor):
        return int(x.shape[0])
    return len(x)


def _take_subset(array, indices: torch.Tensor):
    """Index array-like by 1D long tensor of indices."""
    idx = indices.tolist()
    if isinstance(array, torch.Tensor):
        return array[indices]
    return [array[i] for i in idx]


def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle: bool = True,
    stratify=None,
):
    """Split arrays into random train and test subsets.

    Parameters
    ----------
    *arrays : array-like
        One or more array-likes with the same number of samples
        (torch.Tensor or list). Each is split the same way.
    test_size : float or int, optional
        If float, proportion of samples for the test set in (0, 1).
        If int, absolute number of test samples.
        If None, defaults to 0.25 unless ``train_size`` is given.
    train_size : float or int, optional
        If float, proportion of samples for the train set in (0, 1).
        If int, absolute number of train samples.
        If None, it is set to complement ``test_size``.
    random_state : int, torch.Generator or None, default=None
        Seed or generator used when ``shuffle=True``.
    shuffle : bool, default=True
        If True, shuffle samples before splitting.
    stratify : array-like or None, default=None
        Stratified splitting is not supported yet.

    Returns
    -------
    splitting : list
        For each input array, a (train, test) pair in input order,
        i.e. ``[X_train, X_test, y_train, y_test, ...]``.

    Raises
    ------
    ValueError
        If inputs have inconsistent lengths, sizes are invalid,
        or ``train_size`` + ``test_size`` exceeds the sample count.
    NotImplementedError
        If ``stratify`` is not None.
    TypeError
        If ``random_state`` is neither None, int nor torch.Generator.
    """
    if stratify is not None:
        raise NotImplementedError("Stratified splitting is not supported yet.")
    if len(arrays) == 0:
        raise ValueError("At least one array is required.")
    if not isinstance(shuffle, bool):
        raise TypeError(f"shuffle must be a bool, got {type(shuffle).__name__}.")

    generator = check_random_state(
        random_state,
        arrays[0].device if isinstance(arrays[0], torch.Tensor) else None,
    )

    n_samples = _num_samples(arrays[0])
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}.")
    for i, a in enumerate(arrays[1:], start=1):
        if _num_samples(a) != n_samples:
            raise ValueError(
                "All inputs must have the same number of samples. "
                f"Got {n_samples} and {_num_samples(a)} for inputs 0 and {i}."
            )

    if test_size is None and train_size is None:
        test_size = 0.25

    def _to_n(size, name: str) -> int | None:
        if size is None:
            return None
        if isinstance(size, float):
            if not 0.0 < size < 1.0:
                raise ValueError(f"{name} as float must be in (0, 1), got {size}.")
            return math.ceil(size * n_samples)
        if isinstance(size, int) and not isinstance(size, bool):
            if not 1 <= size < n_samples:
                raise ValueError(
                    f"{name} as int must satisfy 1 <= {name} < n_samples "
                    f"({n_samples}), got {size}."
                )
            return size
        raise TypeError(
            f"{name} must be float in (0, 1), int, or None. "
            f"Got {type(size).__name__}."
        )

    n_test = _to_n(test_size, "test_size")
    n_train = _to_n(train_size, "train_size")
    if n_test is None:
        n_test = n_samples - n_train
    if n_train is None:
        n_train = n_samples - n_test
    if n_train + n_test > n_samples:
        raise ValueError(
            f"train_size ({n_train}) + test_size ({n_test}) must be "
            f"<= n_samples ({n_samples})."
        )
    if n_train < 1 or n_test < 1:
        raise ValueError("Train and test sets must each have at least 1 sample.")

    if shuffle:
        _device = arrays[0].device if isinstance(arrays[0], torch.Tensor) else None
        perm = torch.randperm(n_samples, generator=generator, device=_device)
    else:
        _device = arrays[0].device if isinstance(arrays[0], torch.Tensor) else None
        perm = torch.arange(n_samples, device=_device)
    train_idx = perm[:n_train]
    test_idx = perm[n_train : n_train + n_test]

    out = []
    for a in arrays:
        out.append(_take_subset(a, train_idx))
        out.append(_take_subset(a, test_idx))
    return out


class KFold(BaseEstimator):
    """K-Fold cross-validator.

    Splits samples into ``n_splits`` consecutive folds (shuffled first
    when ``shuffle=True``). Each fold is used once as test set while
    the remaining folds form the train set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        If True, shuffle samples before splitting.
    random_state : int, torch.Generator or None, default=None
        Seed or generator used when ``shuffle=True``. Must be None
        when ``shuffle=False``.
    """

    name = "KFold"

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state=None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of folds.

        Parameters
        ----------
        X : array-like or None
            Ignored, present for API compatibility.
        y : array-like or None
            Ignored, present for API compatibility.
        groups : array-like or None
            Ignored, present for API compatibility.

        Returns
        -------
        n_splits : int
            Number of folds.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate train/test indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...) or (n_samples,)
            Data with ``n_samples`` >= ``n_splits``.
        y : array-like or None, default=None
            Ignored, present for API compatibility.
        groups : array-like or None, default=None
            Ignored, present for API compatibility.

        Yields
        ------
        train_idx : torch.Tensor
            Training indices for the fold.
        test_idx : torch.Tensor
            Test indices for the fold.

        Raises
        ------
        ValueError
            If ``n_splits`` < 2, ``n_samples`` < ``n_splits``,
            or ``random_state`` is set while ``shuffle=False``.
        TypeError
            If hyperparameters have invalid types.
        """
        if not isinstance(self.n_splits, int) or isinstance(self.n_splits, bool):
            raise TypeError(
                "n_splits must be an int, " f"got {type(self.n_splits).__name__}."
            )
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {self.n_splits}.")
        if not isinstance(self.shuffle, bool):
            raise TypeError(
                f"shuffle must be a bool, got {type(self.shuffle).__name__}."
            )
        if not self.shuffle and self.random_state is not None:
            raise ValueError("random_state must be None when shuffle=False.")
        generator = check_random_state(
            self.random_state, X.device if isinstance(X, torch.Tensor) else None
        )

        n_samples = _num_samples(X)
        if n_samples < self.n_splits:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_splits ({self.n_splits})."
            )

        if self.shuffle:
            _device = X.device if isinstance(X, torch.Tensor) else None
            indices = torch.randperm(n_samples, generator=generator, device=_device)
        else:
            _device = X.device if isinstance(X, torch.Tensor) else None
            indices = torch.arange(n_samples, device=_device)

        fold_sizes = [n_samples // self.n_splits] * self.n_splits
        for i in range(n_samples % self.n_splits):
            fold_sizes[i] += 1

        start = 0
        for fold_size in fold_sizes:
            stop = start + fold_size
            test_idx = indices[start:stop]
            train_idx = torch.cat([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            start = stop
