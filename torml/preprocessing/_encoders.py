"""One-hot encoding with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import TransformerMixin
from torml.utils._validation import check_array, check_is_fitted


def _to_2d_list(X) -> list[list]:
    """Convert 2D tensor or nested list to a list of rows."""
    if isinstance(X, torch.Tensor):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D.")
        return [[v.item() for v in list(row)] for row in list(X)]
    if isinstance(X, (list, tuple)):
        rows = list(X)
        if len(rows) == 0:
            raise ValueError("X must contain at least 1 sample.")
        if not all(isinstance(r, (list, tuple)) for r in rows):
            raise ValueError("X must be 2D: a list of rows.")
        width = len(rows[0])
        for r in rows:
            if len(r) != width:
                raise ValueError("All rows in X must have the same length.")
        return [list(r) for r in rows]
    raise TypeError(f"X must be a torch.Tensor or nested list, got {type(X).__name__}.")


class OneHotEncoder(TransformerMixin):
    """Encode categorical features as one-hot numeric arrays.

    Parameters
    ----------
    categories : 'auto' or list of array-likes, default='auto'
        Categories per feature. ``'auto'`` learns sorted unique values
        from the training set.
    handle_unknown : {'error', 'ignore'}, default='error'
        How to handle unknown categories in :meth:`transform`.
        ``'ignore'`` encodes them as all zeros.
    sparse_output : bool, default=False
        Only dense output is supported; ``True`` raises an error.
    dtype : torch.dtype or str, default=torch.float32
        Output dtype.

    Attributes
    ----------
    categories_ : list of lists
        Categories per feature learned or provided during fit.
    n_features_in_ : int
        Number of features seen during fit.
    n_features_out_ : int
        Total one-hot output width.
    """

    name = "OneHotEncoder"

    def __init__(
        self,
        categories="auto",
        handle_unknown="error",
        sparse_output=False,
        dtype=torch.float32,
    ):
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.dtype = dtype

    def _resolve_dtype(self) -> torch.dtype:
        if isinstance(self.dtype, torch.dtype):
            return self.dtype
        if isinstance(self.dtype, str):
            try:
                return getattr(torch, self.dtype)
            except AttributeError as e:
                raise ValueError(f"Unknown dtype string {self.dtype!r}.") from e
        raise TypeError(
            f"dtype must be a torch.dtype or string, "
            f"got {type(self.dtype).__name__}."
        )

    def fit(self, X, y=None):
        """Learn categories per feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Categorical training data as a 2D tensor or nested list.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : OneHotEncoder
            Fitted encoder.
        """
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(
                "handle_unknown must be 'error' or 'ignore', "
                f"got {self.handle_unknown!r}."
            )
        if self.sparse_output not in (False,):
            raise ValueError(
                "Only dense output is supported; sparse_output must be False."
            )
        rows = _to_2d_list(X)
        n_features = len(rows[0])
        self.n_features_in_ = n_features
        if self.categories == "auto":
            cats: list[list] = []
            for j in range(n_features):
                try:
                    uniq = sorted({row[j] for row in rows})
                except TypeError as e:
                    raise TypeError("Categories must be sortable.") from e
                if len(uniq) == 0:
                    raise ValueError("Each feature must have >= 1 category.")
                cats.append(uniq)
            self.categories_ = cats
        elif isinstance(self.categories, (list, tuple)):
            if len(self.categories) != n_features:
                raise ValueError(
                    f"categories has {len(self.categories)} features, "
                    f"but X has {n_features}."
                )
            cats = []
            for c in self.categories:
                if isinstance(c, torch.Tensor):
                    cats.append([v.item() for v in list(c.reshape(-1))])
                else:
                    cats.append(list(c))
            self.categories_ = cats
        else:
            raise TypeError(
                "categories must be 'auto' or a list of array-likes, "
                f"got {self.categories!r}."
            )
        self._cat_index = [
            {c: i for i, c in enumerate(col)} for col in self.categories_
        ]
        self._widths = [len(col) for col in self.categories_]
        self.n_features_out_ = int(sum(self._widths))
        self._out_dtype = self._resolve_dtype()
        return self

    def transform(self, X):
        """One-hot encode ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to encode.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, n_features_out_)
            One-hot encoded data.
        """
        check_is_fitted(self, attributes=["categories_"])
        rows = _to_2d_list(X)
        if len(rows[0]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {len(rows[0])} features, but OneHotEncoder was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        n_samples = len(rows)
        out = torch.zeros(n_samples, int(self.n_features_out_), dtype=self._out_dtype)
        offset = 0
        for j, width in enumerate(self._widths):
            mapping = self._cat_index[j]
            for i, row in enumerate(rows):
                v = row[j]
                key = v.item() if isinstance(v, torch.Tensor) else v
                if key not in mapping:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Unknown category {v!r} in feature {j}.")
                    continue
                out[i, offset + mapping[key]] = 1
            offset += width
        return out

    def _transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        """Decode one-hot data back to categories.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_out_)
            One-hot encoded data.

        Returns
        -------
        X_orig : list of lists
            Decoded categorical rows.
        """
        check_is_fitted(self, attributes=["categories_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        if int(Xt.shape[1]) != int(self.n_features_out_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} columns, but OneHotEncoder output "
                f"has {int(self.n_features_out_)}."
            )
        rows: list[list] = []
        offset = 0
        col_blocks = []
        for width in self._widths:
            col_blocks.append(Xt[:, offset : offset + width])
            offset += width
        for i in range(int(Xt.shape[0])):
            row = []
            for j, block in enumerate(col_blocks):
                k = int(torch.argmax(block[i]).item())
                if float(block[i, k].item()) == 0 and self.handle_unknown == "ignore":
                    row.append(None)
                else:
                    row.append(self.categories_[j][k])
            rows.append(row)
        return rows
