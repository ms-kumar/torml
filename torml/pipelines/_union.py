"""Feature union and column transformers with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator, TransformerMixin, clone
from torml.pipelines._pipeline import _validate_steps
from torml.utils._validation import check_array, check_is_fitted


class FeatureUnion(TransformerMixin):
    """Concatenate outputs of several transformers.

    Parameters
    ----------
    transformer_list : list of (str, transformer)
        Transformers to fit in parallel and concatenate.

    Attributes
    ----------
    transformer_list_ : list of (str, transformer)
        Fitted transformer clones.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "FeatureUnion"

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        """Fit all transformers on ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like or None, default=None
            Targets.

        Returns
        -------
        self : FeatureUnion
            Fitted union.
        """
        members = _validate_steps(self.transformer_list)
        fitted = []
        for name, est in members:
            if not hasattr(est, "transform"):
                raise TypeError(f"Member {name!r} must implement transform.")
            est = clone(est)
            if y is not None:
                est.fit(X, y)
            else:
                est.fit(X)
            fitted.append((name, est))
        self.transformer_list_ = fitted
        arr = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
        self.n_features_in_ = int(arr.shape[1]) if arr.ndim == 2 else 0
        return self

    def transform(self, X):
        """Concatenate member outputs for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, sum of outputs)
            Concatenated features.
        """
        check_is_fitted(self, attributes=["transformer_list_"])
        parts = [est.transform(X) for _, est in self.transformer_list_]
        return torch.cat(
            [p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in parts],
            dim=1,
        )

    def _transform(self, X):
        return self.transform(X)


def _select_columns(X, cols, n_features: int) -> torch.Tensor:
    """Select columns by int, slice, or list of ints."""
    Xt = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
    if Xt.ndim != 2:
        raise ValueError(f"X must be 2D, got {Xt.ndim}D.")
    if isinstance(cols, slice):
        return Xt[:, cols]
    if isinstance(cols, int) and not isinstance(cols, bool):
        if not 0 <= cols < n_features:
            raise ValueError(f"Column {cols} out of range for {n_features} features.")
        return Xt[:, cols : cols + 1]
    if isinstance(cols, (list, tuple)) and all(
        isinstance(c, int) and not isinstance(c, bool) for c in cols
    ):
        if any(not 0 <= c < n_features for c in cols):
            raise ValueError(f"Column indices out of range for {n_features} features.")
        return Xt[:, list(cols)]
    raise TypeError("cols must be int, slice, or list of ints.")


class ColumnTransformer(TransformerMixin):
    """Apply transformers to column subsets and concatenate.

    Parameters
    ----------
    transformers : list of (str, transformer, cols)
        Triples of name, transformer, and column selector (int, slice,
        or list of ints).
    remainder : {'drop', 'passthrough'}, default='drop'
        How to handle unselected columns.

    Attributes
    ----------
    transformers_ : list of (str, transformer, cols)
        Fitted triples.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "ColumnTransformer"

    def __init__(self, transformers, remainder="drop"):
        self.transformers = transformers
        self.remainder = remainder

    def _validate(self):
        """Validate configuration and return cleaned triples."""
        if self.remainder not in ("drop", "passthrough"):
            raise ValueError(
                f"remainder must be 'drop' or 'passthrough', got {self.remainder!r}."
            )
        if not isinstance(self.transformers, (list, tuple)):
            raise TypeError("transformers must be a list of (name, est, cols).")
        cleaned = []
        seen = set()
        for item in self.transformers:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                raise TypeError(
                    "Each entry must be a (name, transformer, cols) triple."
                )
            name, est, cols = item
            if not isinstance(name, str):
                raise TypeError(f"Name must be a str, got {type(name).__name__}.")
            if name in seen:
                raise ValueError(f"Duplicate transformer name {name!r}.")
            if not isinstance(est, BaseEstimator):
                raise TypeError(
                    f"Member {name!r} must be a BaseEstimator, "
                    f"got {type(est).__name__}."
                )
            if not hasattr(est, "transform"):
                raise TypeError(f"Member {name!r} must implement transform.")
            seen.add(name)
            cleaned.append((name, est, cols))
        return cleaned

    def fit(self, X, y=None):
        """Fit members on their column subsets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like or None, default=None
            Targets.

        Returns
        -------
        self : ColumnTransformer
            Fitted transformer.
        """
        cleaned = self._validate()
        arr = check_array(X, ensure_2d=True, dtype=torch.float32)
        n_features = int(arr.shape[1])
        self.n_features_in_ = n_features
        fitted = []
        for name, est, cols in cleaned:
            sub = _select_columns(arr, cols, n_features)
            est = clone(est)
            if y is not None:
                est.fit(sub, y)
            else:
                est.fit(sub)
            fitted.append((name, est, cols))
        self.transformers_ = fitted
        return self

    def transform(self, X):
        """Transform column subsets and concatenate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        Xt : torch.Tensor
            Concatenated outputs.
        """
        check_is_fitted(self, attributes=["transformers_"])
        arr = check_array(X, ensure_2d=True, dtype=torch.float32)
        if int(arr.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(arr.shape[1])} features, but ColumnTransformer "
                f"was fitted with {int(self.n_features_in_)} features."
            )
        parts: list = []
        used: set = set()
        for _, est, cols in self.transformers_:
            sub = _select_columns(arr, cols, self.n_features_in_)
            parts.append(est.transform(sub))
            if isinstance(cols, slice):
                used.update(range(*cols.indices(self.n_features_in_)))
            elif isinstance(cols, int):
                used.add(cols)
            else:
                used.update(cols)
        if self.remainder == "passthrough":
            rest = [j for j in range(self.n_features_in_) if j not in used]
            if rest:
                parts.append(arr[:, rest])
        if not parts:
            raise ValueError("No output columns selected.")
        return torch.cat(
            [p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in parts],
            dim=1,
        )

    def _transform(self, X):
        return self.transform(X)
