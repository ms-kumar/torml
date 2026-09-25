"""Label encoding with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator
from torml.utils._validation import check_is_fitted


def _as_list(y) -> list:
    """Convert tensor or list input to a Python list of labels."""
    if isinstance(y, torch.Tensor):
        if y.ndim == 0:
            return [y.item()]
        return [v.item() if isinstance(v, torch.Tensor) else v for v in list(y)]
    if isinstance(y, (list, tuple)):
        return list(y)
    raise TypeError(f"y must be a torch.Tensor, list or tuple, got {type(y).__name__}.")


class LabelEncoder(BaseEstimator):
    """Encode target labels with values between 0 and n_classes - 1.

    Attributes
    ----------
    classes_ : list or torch.Tensor
        Sorted unique labels seen during fit. A tensor when fitted on a
        tensor, otherwise a sorted list.
    n_features_in_ : int
        Always 1 for 1D label input (set for API compatibility).
    """

    name = "LabelEncoder"

    def __init__(self):
        pass

    def fit(self, y):
        """Fit the encoder to ``y``.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target labels as a tensor, list or tuple.

        Returns
        -------
        self : LabelEncoder
            Fitted encoder.
        """
        values = _as_list(y)
        if len(values) == 0:
            raise ValueError("y must contain at least 1 label.")
        if isinstance(y, torch.Tensor):
            flat = y.reshape(-1)
            self.classes_ = torch.unique(flat, sorted=True)
            self._classes_list = [v.item() for v in list(self.classes_)]
        else:
            try:
                uniq = sorted(set(values))
            except TypeError as e:
                raise TypeError("Labels must be sortable.") from e
            self.classes_ = uniq
            self._classes_list = list(uniq)
        self._class_to_index = {c: i for i, c in enumerate(self._classes_list)}
        self.n_features_in_ = 1
        return self

    def transform(self, y):
        """Encode ``y`` as integers.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Labels to encode.

        Returns
        -------
        encoded : torch.Tensor of shape (n_samples,)
            Integer class indices.
        """
        check_is_fitted(self, attributes=["classes_"])
        values = _as_list(y)
        out = []
        for v in values:
            key = v.item() if isinstance(v, torch.Tensor) else v
            if key not in self._class_to_index:
                raise ValueError(f"Unknown label {v!r} seen in transform.")
            out.append(self._class_to_index[key])
        device = y.device if isinstance(y, torch.Tensor) else None
        kwargs = {"device": device} if device is not None else {}
        return torch.tensor(out, dtype=torch.long, **kwargs)

    def fit_transform(self, y):
        """Fit and encode ``y``.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        encoded : torch.Tensor of shape (n_samples,)
            Integer class indices.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        """Decode integer indices back to original labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Integer class indices.

        Returns
        -------
        labels : torch.Tensor or list
            Original labels: a tensor when fitted on a tensor, else a list.
        """
        check_is_fitted(self, attributes=["classes_"])
        if isinstance(y, torch.Tensor):
            idx = y.reshape(-1).tolist()
        elif isinstance(y, (list, tuple)):
            idx = list(y)
        else:
            raise TypeError(
                "y must be a torch.Tensor, list or tuple, " f"got {type(y).__name__}."
            )
        n_classes = len(self._classes_list)
        decoded = []
        for i in idx:
            ii = int(i.item() if isinstance(i, torch.Tensor) else i)
            if not 0 <= ii < n_classes:
                raise ValueError(
                    f"Invalid class index {ii}; must be in [0, {n_classes})."
                )
            decoded.append(self._classes_list[ii])
        if isinstance(self.classes_, torch.Tensor):
            device = y.device if isinstance(y, torch.Tensor) else None
            kwargs = {"device": device} if device is not None else {}
            return torch.as_tensor(decoded, dtype=self.classes_.dtype, **kwargs)
        return decoded
