"""Exhaustive grid search with a PyTorch backend."""

from __future__ import annotations

import itertools

import torch

from torml.base import BaseEstimator, clone
from torml.model_selection._validation import cross_val_score
from torml.utils._validation import check_is_fitted


def _expand_grid(param_grid) -> tuple[list[str], list[tuple]]:
    """Normalize ``param_grid`` to (keys, combinations)."""
    if isinstance(param_grid, dict):
        grids = [param_grid]
    elif isinstance(param_grid, (list, tuple)):
        grids = list(param_grid)
    else:
        raise TypeError("param_grid must be a dict or list of dicts.")
    keys: list[str] = []
    combos: list[tuple] = []
    for grid in grids:
        if not isinstance(grid, dict) or len(grid) == 0:
            raise ValueError("Each param grid must be a non-empty dict.")
        sub_keys = sorted(grid.keys())
        values = []
        for key in sub_keys:
            vals = grid[key]
            if isinstance(vals, str) or not isinstance(vals, (list, tuple)):
                raise TypeError(
                    f"Values for {key!r} must be a list or tuple, "
                    f"got {type(vals).__name__}."
                )
            if len(vals) == 0:
                raise ValueError(f"Values for {key!r} must be non-empty.")
            values.append(list(vals))
        for combo in itertools.product(*values):
            keys = sub_keys
            combos.append(combo)
    if not combos:
        raise ValueError("param_grid produced no combinations.")
    return keys, combos


class GridSearchCV(BaseEstimator):
    """Exhaustive search over a parameter grid with cross-validation.

    Parameters
    ----------
    estimator : estimator
        Estimator supporting ``get_params``/``set_params``/``fit``.
    param_grid : dict or list of dicts
        Parameter names mapped to lists of values to try.
    cv : int or splitter, default=5
        Cross-validation splitting (see :func:`cross_val_score`).
    scoring : str, callable or None, default=None
        Scoring (see :func:`cross_val_score`).
    refit : bool, default=True
        Refit the best estimator on the full data.

    Attributes
    ----------
    cv_results_ : dict with ``params`` and ``mean_test_score`` lists.
    best_params_ : dict
        Best combination found.
    best_score_ : float
        Its mean CV score.
    best_estimator_ : estimator
        Best estimator, refit on the full data when ``refit=True``.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "GridSearchCV"

    def __init__(self, estimator, param_grid, cv=5, scoring=None, refit=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

    def fit(self, X, y=None):
        """Evaluate all combinations and optionally refit the best.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or None, default=None
            Targets.

        Returns
        -------
        self : GridSearchCV
            Fitted search.
        """
        if not isinstance(self.estimator, BaseEstimator):
            raise TypeError(
                "estimator must be a BaseEstimator, "
                f"got {type(self.estimator).__name__}."
            )
        if not isinstance(self.refit, bool):
            raise TypeError(f"refit must be a bool, got {type(self.refit).__name__}.")
        keys, combos = _expand_grid(self.param_grid)
        Xt = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
        self.n_features_in_ = int(Xt.shape[1]) if Xt.ndim == 2 else 0
        params_list, means = [], []
        for combo in combos:
            params = dict(zip(keys, combo))
            est = clone(self.estimator).set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            params_list.append(params)
            means.append(float(scores.mean()))
        order = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
        self.cv_results_ = {
            "params": [params_list[i] for i in order],
            "mean_test_score": [means[i] for i in order],
        }
        self.best_params_ = dict(self.cv_results_["params"][0])
        self.best_score_ = float(self.cv_results_["mean_test_score"][0])
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        if self.refit:
            if y is None:
                self.best_estimator_.fit(X)
            else:
                self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        """Predict with the best estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor
            Best estimator predictions.
        """
        check_is_fitted(self, attributes=["best_estimator_"])
        if not hasattr(self.best_estimator_, "predict"):
            raise AttributeError("Best estimator does not implement predict.")
        return self.best_estimator_.predict(X)

    def score(self, X, y=None):
        """Score with the best estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.
        y : array-like or None, default=None
            Targets.

        Returns
        -------
        score : float
            Best estimator score.
        """
        check_is_fitted(self, attributes=["best_estimator_"])
        if y is None:
            return self.best_estimator_.score(X)
        return self.best_estimator_.score(X, y)
