"""Scaling transformers with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import TransformerMixin
from torml.utils._validation import check_array, check_is_fitted


class StandardScaler(TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.

    Attributes
    ----------
    mean_ : torch.Tensor of shape (n_features,)
        Per-feature mean estimated from the training set.
    var_ : torch.Tensor of shape (n_features,)
        Per-feature variance estimated from the training set.
    scale_ : torch.Tensor of shape (n_features,)
        Per-feature standard deviation used for scaling. Ones when
        ``with_std=False`` or variance is zero.
    n_features_in_ : int
        Number of features seen during fit.
    n_samples_seen_ : int
        Number of samples seen during fit.
    """

    name = "StandardScaler"

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        """Compute the mean and variance for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : StandardScaler
            Fitted scaler.
        """
        if not isinstance(self.with_mean, bool):
            raise TypeError(
                "with_mean must be a bool, " f"got {type(self.with_mean).__name__}."
            )
        if not isinstance(self.with_std, bool):
            raise TypeError(
                f"with_std must be a bool, got {type(self.with_std).__name__}."
            )
        Xt = check_array(X, ensure_2d=True)
        self.n_features_in_ = int(Xt.shape[1])
        self.n_samples_seen_ = int(Xt.shape[0])
        if self.with_mean:
            self.mean_ = Xt.mean(dim=0)
        else:
            self.mean_ = torch.zeros(
                self.n_features_in_, dtype=Xt.dtype, device=Xt.device
            )
        if self.with_std:
            self.var_ = Xt.var(dim=0, unbiased=False)
            scale = torch.sqrt(self.var_)
            scale[scale == 0] = 1.0
            self.scale_ = scale
        else:
            self.var_ = torch.zeros(
                self.n_features_in_, dtype=Xt.dtype, device=Xt.device
            )
            self.scale_ = torch.ones(
                self.n_features_in_, dtype=Xt.dtype, device=Xt.device
            )
        return self

    def transform(self, X):
        """Standardize X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self, attributes=["mean_", "scale_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but StandardScaler was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        out = Xt.to(dtype=self.mean_.dtype)
        if self.with_mean:
            out = out - self.mean_
        if self.with_std:
            out = out / self.scale_
        return out

    def _transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        """Undo the standardization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Standardized data.

        Returns
        -------
        X_orig : torch.Tensor of shape (n_samples, n_features)
            Data in the original space.
        """
        check_is_fitted(self, attributes=["mean_", "scale_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but StandardScaler was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        out = Xt.to(dtype=self.mean_.dtype)
        if self.with_std:
            out = out * self.scale_
        if self.with_mean:
            out = out + self.mean_
        return out


class MinMaxScaler(TransformerMixin):
    """Scale features to a given range.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data. Must satisfy min < max.

    Attributes
    ----------
    data_min_ : torch.Tensor of shape (n_features,)
        Per-feature minimum seen during fit.
    data_max_ : torch.Tensor of shape (n_features,)
        Per-feature maximum seen during fit.
    data_range_ : torch.Tensor of shape (n_features,)
        Per-feature range (max - min). Ones where the range is zero.
    scale_ : torch.Tensor of shape (n_features,)
        Per-feature scaling factor.
    min_ : torch.Tensor of shape (n_features,)
        Per-feature offset.
    n_features_in_ : int
        Number of features seen during fit.
    n_samples_seen_ : int
        Number of samples seen during fit.
    """

    name = "MinMaxScaler"

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def _validate_range(self):
        fr = self.feature_range
        if (
            not isinstance(fr, (tuple, list))
            or len(fr) != 2
            or not all(isinstance(v, (int, float)) for v in fr)
        ):
            raise TypeError(
                "feature_range must be a tuple (min, max) of numbers, " f"got {fr!r}."
            )
        lo, hi = float(fr[0]), float(fr[1])
        if lo >= hi:
            raise ValueError(f"feature_range min must be < max, got ({lo}, {hi}).")
        return lo, hi

    def fit(self, X, y=None):
        """Compute the min and max for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : MinMaxScaler
            Fitted scaler.
        """
        lo, hi = self._validate_range()
        Xt = check_array(X, ensure_2d=True)
        self.n_features_in_ = int(Xt.shape[1])
        self.n_samples_seen_ = int(Xt.shape[0])
        self.data_min_ = Xt.min(dim=0).values
        self.data_max_ = Xt.max(dim=0).values
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0
        self.data_range_ = data_range
        self.scale_ = (hi - lo) / self.data_range_
        self.min_ = lo - self.data_min_ * self.scale_
        self._feature_lo = lo
        self._feature_hi = hi
        return self

    def transform(self, X):
        """Scale X to ``feature_range``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self, attributes=["scale_", "min_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but MinMaxScaler was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        return Xt * self.scale_ + self.min_

    def _transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        """Undo the scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled data.

        Returns
        -------
        X_orig : torch.Tensor of shape (n_samples, n_features)
            Data in the original space.
        """
        check_is_fitted(self, attributes=["scale_", "min_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but MinMaxScaler was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        return (Xt - self.min_) / self.scale_
