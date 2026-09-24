"""Dictionary vectorization with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import TransformerMixin
from torml.utils._validation import check_is_fitted


class DictVectorizer(TransformerMixin):
    """Convert string-keyed dicts to a numeric matrix.

    Parameters
    ----------
    sort : bool, default=True
        Sort feature names alphabetically. If False, use first-seen order.

    Attributes
    ----------
    vocabulary_ : dict of str to int
        Feature name to column mapping.
    feature_names_ : list of str
        Ordered feature names.
    n_features_in_ : int
        Number of features (columns) learned.
    """

    name = "DictVectorizer"

    def __init__(self, sort=True):
        self.sort = sort

    def fit(self, X, y=None):
        """Learn the vocabulary from dict rows.

        Parameters
        ----------
        X : iterable of dicts
            Rows mapping feature names to numbers.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : DictVectorizer
            Fitted vectorizer.
        """
        rows = list(X)
        if len(rows) == 0:
            raise ValueError("X must contain at least 1 row.")
        names: list = []
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(f"Rows must be dicts, got {type(row).__name__}.")
            for key, value in row.items():
                if not isinstance(key, str):
                    raise TypeError(f"Keys must be str, got {type(key).__name__}.")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(
                        f"Values must be numbers, got {type(value).__name__}."
                    )
                if key not in names:
                    names.append(key)
        if self.sort:
            names = sorted(names)
        self.feature_names_ = names
        self.vocabulary_ = {name: i for i, name in enumerate(names)}
        self.n_features_in_ = len(names)
        return self

    def transform(self, X):
        """Vectorize dict rows.

        Parameters
        ----------
        X : iterable of dicts
            Rows to vectorize.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, n_features)
            Numeric matrix.
        """
        check_is_fitted(self, attributes=["vocabulary_"])
        rows = list(X)
        out = torch.zeros(len(rows), len(self.feature_names_), dtype=torch.float32)
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"Rows must be dicts, got {type(row).__name__}.")
            for key, value in row.items():
                if key not in self.vocabulary_:
                    raise ValueError(f"Unknown feature {key!r} in transform.")
                out[i, self.vocabulary_[key]] = float(value)
        return out

    def _transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        """Convert a matrix back to dict rows.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric matrix.

        Returns
        -------
        rows : list of dicts
            Nonzero entries per row.
        """
        check_is_fitted(self, attributes=["vocabulary_"])
        Xt = torch.as_tensor(X, dtype=torch.float32)
        if int(Xt.shape[1]) != len(self.feature_names_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} columns, but the vocabulary has "
                f"{len(self.feature_names_)}."
            )
        rows = []
        for i in range(int(Xt.shape[0])):
            row = {}
            for j in torch.where(Xt[i] != 0)[0].tolist():
                row[self.feature_names_[j]] = float(Xt[i, j].item())
            rows.append(row)
        return rows

    def get_feature_names_out(self):
        """Return ordered feature names.

        Returns
        -------
        names : list of str
            Feature names.
        """
        check_is_fitted(self, attributes=["feature_names_"])
        return list(self.feature_names_)
