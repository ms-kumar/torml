"""Multi-output meta-estimators with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from torml.utils._validation import check_array, check_is_fitted


def _validate_2d_y(y, name: str) -> torch.Tensor:
    """Validate a 2D multi-output target matrix."""
    arr = torch.as_tensor(y) if not isinstance(y, torch.Tensor) else y
    if arr.ndim == 1:
        arr = arr.unsqueeze(1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {arr.ndim}D.")
    return arr


class MultiOutputRegressor(RegressorMixin):
    """One regressor clone per output column.

    Parameters
    ----------
    estimator : estimator
        Regressor supporting ``fit``/``predict``.

    Attributes
    ----------
    estimators_ : list of fitted regressor clones.
    n_features_in_ : int
        Number of features seen during fit.
    n_outputs_ : int
        Number of target columns.
    """

    name = "MultiOutputRegressor"

    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        """Get parameters including ``estimator__param`` entries."""
        params = {"estimator": self.estimator}
        if deep and isinstance(self.estimator, BaseEstimator):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        """Set parameters including ``estimator__param`` entries."""
        nested = {}
        for key, value in params.items():
            if key == "estimator":
                self.estimator = value
            elif key.startswith("estimator__"):
                nested[key.split("__", 1)[1]] = value
            else:
                raise ValueError(f"Invalid parameter {key!r} for MultiOutputRegressor.")
        if nested:
            self.estimator = clone(self.estimator).set_params(**nested)
        return self

    def fit(self, X, y):
        """Fit one clone per target column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Targets.

        Returns
        -------
        self : MultiOutputRegressor
            Fitted wrapper.
        """
        if not isinstance(self.estimator, BaseEstimator):
            raise TypeError(
                "estimator must be a BaseEstimator, "
                f"got {type(self.estimator).__name__}."
            )
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        Yt = _validate_2d_y(y, "y").to(dtype=torch.float32)
        if int(Xt.shape[0]) != int(Yt.shape[0]):
            raise ValueError(
                f"X and y have inconsistent lengths: {int(Xt.shape[0])} "
                f"vs {int(Yt.shape[0])}."
            )
        self.n_features_in_ = int(Xt.shape[1])
        self.n_outputs_ = int(Yt.shape[1])
        self.estimators_ = [
            clone(self.estimator).fit(Xt, Yt[:, j]) for j in range(self.n_outputs_)
        ]
        return self

    def predict(self, X):
        """Predict all outputs for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples, n_outputs)
            Predictions per output.
        """
        check_is_fitted(self, attributes=["estimators_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        cols = [
            torch.as_tensor(est.predict(Xt), dtype=torch.float32).reshape(-1)
            for est in self.estimators_
        ]
        stacked = torch.stack(cols, dim=1)
        return stacked.squeeze(1) if stacked.shape[1] == 1 else stacked


class MultiOutputClassifier(ClassifierMixin):
    """One classifier clone per output column.

    Parameters
    ----------
    estimator : estimator
        Classifier supporting ``fit``/``predict``.

    Attributes
    ----------
    estimators_ : list of fitted classifier clones.
    classes_ : list per output of class labels.
    n_features_in_ : int
        Number of features seen during fit.
    n_outputs_ : int
        Number of target columns.
    """

    name = "MultiOutputClassifier"

    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        """Get parameters including ``estimator__param`` entries."""
        params = {"estimator": self.estimator}
        if deep and isinstance(self.estimator, BaseEstimator):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        """Set parameters including ``estimator__param`` entries."""
        nested = {}
        for key, value in params.items():
            if key == "estimator":
                self.estimator = value
            elif key.startswith("estimator__"):
                nested[key.split("__", 1)[1]] = value
            else:
                raise ValueError(
                    f"Invalid parameter {key!r} for MultiOutputClassifier."
                )
        if nested:
            self.estimator = clone(self.estimator).set_params(**nested)
        return self

    def fit(self, X, y):
        """Fit one clone per target column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Class labels per output.

        Returns
        -------
        self : MultiOutputClassifier
            Fitted wrapper.
        """
        if not isinstance(self.estimator, BaseEstimator):
            raise TypeError(
                "estimator must be a BaseEstimator, "
                f"got {type(self.estimator).__name__}."
            )
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        Yt = _validate_2d_y(y, "y")
        if int(Xt.shape[0]) != int(Yt.shape[0]):
            raise ValueError(
                f"X and y have inconsistent lengths: {int(Xt.shape[0])} "
                f"vs {int(Yt.shape[0])}."
            )
        self.n_features_in_ = int(Xt.shape[1])
        self.n_outputs_ = int(Yt.shape[1])
        self.estimators_ = []
        self.classes_ = []
        for j in range(self.n_outputs_):
            est = clone(self.estimator).fit(Xt, Yt[:, j])
            self.estimators_.append(est)
            self.classes_.append(est.classes_ if hasattr(est, "classes_") else None)
        return self

    def predict(self, X):
        """Predict all outputs for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples, n_outputs)
            Predicted labels per output.
        """
        check_is_fitted(self, attributes=["estimators_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        cols = []
        for est in self.estimators_:
            pred = est.predict(Xt)
            vals = pred.tolist() if isinstance(pred, torch.Tensor) else list(pred)
            cols.append(torch.as_tensor(vals))
        stacked = torch.stack(cols, dim=1)
        return stacked
