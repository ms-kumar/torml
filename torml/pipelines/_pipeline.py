"""Pipeline chaining transforms with a final estimator."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator, clone
from torml.utils._validation import check_is_fitted


def _validate_steps(steps) -> list:
    """Validate ``steps`` as a non-empty list of (name, estimator) pairs."""
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise TypeError("steps must be a non-empty list of (name, estimator).")
    cleaned = []
    seen = set()
    for item in steps:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise TypeError("Each step must be a (name, estimator) pair.")
        name, est = item
        if not isinstance(name, str):
            raise TypeError(f"Step name must be a str, got {type(name).__name__}.")
        if name in seen:
            raise ValueError(f"Duplicate step name {name!r}.")
        if not isinstance(est, BaseEstimator):
            raise TypeError(
                f"Step {name!r} must be a BaseEstimator, got {type(est).__name__}."
            )
        seen.add(name)
        cleaned.append((name, est))
    return cleaned


class Pipeline(BaseEstimator):
    """Chain transforms with a final estimator.

    Intermediate steps must implement ``fit``/``transform``; the last step
    implements ``fit`` plus ``predict`` and/or ``transform``/``score``.

    Parameters
    ----------
    steps : list of (str, estimator)
        Ordered chain of transforms and a final estimator.

    Attributes
    ----------
    steps_ : list of (str, estimator)
        Fitted step clones.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "Pipeline"

    def __init__(self, steps):
        self.steps = steps

    def get_params(self, deep=True):
        """Get parameters including per-step ``name__param`` entries."""
        params = {"steps": self.steps}
        if deep:
            for name, est in _validate_steps(self.steps):
                for k, v in est.get_params(deep=True).items():
                    params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params):
        """Set parameters including per-step ``name__param`` entries."""
        nested: dict = {}
        for key, value in params.items():
            if "__" in key:
                name, sub = key.split("__", 1)
                nested.setdefault(name, {})[sub] = value
            elif key == "steps":
                self.steps = value
            else:
                raise ValueError(f"Invalid parameter {key!r} for Pipeline.")
        if nested:
            members = dict(_validate_steps(self.steps))
            rebuilt = []
            for name, est in members.items():
                if name in nested:
                    est = clone(est).set_params(**nested[name])
                rebuilt.append((name, est))
            self.steps = rebuilt
        return self

    @property
    def _final(self):
        """Return the final fitted step."""
        return self.steps_[-1][1]

    def fit(self, X, y=None):
        """Fit transforms in sequence, then the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or None, default=None
            Targets.

        Returns
        -------
        self : Pipeline
            Fitted pipeline.
        """
        steps = _validate_steps(self.steps)
        Xt, yt = X, y
        fitted = []
        for i, (name, est) in enumerate(steps):
            est = clone(est)
            last = i == len(steps) - 1
            if yt is not None:
                est.fit(Xt, yt)
            else:
                est.fit(Xt)
            if not last:
                if not hasattr(est, "transform"):
                    raise TypeError(
                        f"Intermediate step {name!r} must implement transform."
                    )
                Xt = est.transform(Xt)
            fitted.append((name, est))
        self.steps_ = fitted
        arr = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
        self.n_features_in_ = int(arr.shape[1]) if arr.ndim == 2 else 0
        return self

    def _route_transform(self, X):
        """Apply intermediate transforms to ``X``."""
        Xt = X
        for _, est in self.steps_[:-1]:
            Xt = est.transform(Xt)
        return Xt

    def predict(self, X):
        """Predict with the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor
            Final estimator predictions.
        """
        check_is_fitted(self, attributes=["steps_"])
        final = self._final
        if not hasattr(final, "predict"):
            raise AttributeError("Final step does not implement predict.")
        return final.predict(self._route_transform(X))

    def predict_proba(self, X):
        """Predict probabilities with the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        proba : torch.Tensor
            Final estimator probabilities.
        """
        check_is_fitted(self, attributes=["steps_"])
        final = self._final
        if not hasattr(final, "predict_proba"):
            raise AttributeError("Final step does not implement predict_proba.")
        return final.predict_proba(self._route_transform(X))

    def transform(self, X):
        """Transform ``X`` through all transforming steps.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        Xt : torch.Tensor
            Transformed data.
        """
        check_is_fitted(self, attributes=["steps_"])
        Xt = X
        for _, est in self.steps_:
            if not hasattr(est, "transform"):
                raise AttributeError("A step does not implement transform.")
            Xt = est.transform(Xt)
        return Xt

    def score(self, X, y=None):
        """Score with the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.
        y : array-like or None, default=None
            Targets.

        Returns
        -------
        score : float
            Final estimator score.
        """
        check_is_fitted(self, attributes=["steps_"])
        final = self._final
        if not hasattr(final, "score"):
            raise AttributeError("Final step does not implement score.")
        Xt = self._route_transform(X)
        return final.score(Xt, y) if y is not None else final.score(Xt)
