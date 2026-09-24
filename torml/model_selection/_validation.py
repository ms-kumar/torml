"""Validation utilities for model selection.

Provides :func:`cross_val_score` with a PyTorch backend.
"""

from __future__ import annotations

import torch

from torml.base import BaseEstimator, clone


def _resolve_scorer(scoring):
    """Return a ``(y_true, y_pred) -> float`` function for a scoring string."""
    if scoring in ("accuracy", "accuracy_score"):
        from torml.metrics import accuracy_score

        return accuracy_score
    if scoring in ("r2", "r2_score"):
        from torml.metrics import r2_score

        return r2_score
    if scoring in ("neg_mean_squared_error", "neg_mse"):
        from torml.metrics import mean_squared_error

        return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    raise ValueError(
        "Unknown scoring string "
        f"{scoring!r}. Supported: 'accuracy', 'r2', 'neg_mean_squared_error', "
        "a callable, or None."
    )


def cross_val_score(estimator, X, y=None, *, cv=5, scoring=None):
    """Evaluate an estimator with cross-validation.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator implementing ``fit`` and (``predict`` or ``score``).
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or None, default=None
        Target values.
    cv : int or splitter, default=5
        If int, number of folds for :class:`KFold` (no shuffling).
        Otherwise an object with a ``split(X, y)`` method yielding
        ``(train_idx, test_idx)`` pairs.
    scoring : str, callable or None, default=None
        If None, use ``estimator.score``. If str, one of ``'accuracy'``,
        ``'r2'``, ``'neg_mean_squared_error'``. If callable with signature
        ``scoring(estimator, X_test, y_test)``, its return value is used;
        if it instead accepts ``(y_true, y_pred)``, predictions from
        ``estimator.predict`` are passed.

    Returns
    -------
    scores : torch.Tensor of shape (n_splits,)
        Score for each fold.

    Raises
    ------
    ValueError
        If ``cv`` is invalid or train/test indexing fails.
    TypeError
        If ``estimator`` is not a :class:`BaseEstimator` or ``cv``
        has an unsupported type.
    """
    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            "estimator must be a BaseEstimator instance, "
            f"got {type(estimator).__name__}."
        )
    if not hasattr(estimator, "fit"):
        raise TypeError("estimator must implement fit.")
    if scoring is not None and not (isinstance(scoring, str) or callable(scoring)):
        raise TypeError("scoring must be a string, callable, or None.")

    if isinstance(cv, int):
        if cv < 2:
            raise ValueError(f"cv must be at least 2, got {cv}.")
        from torml.model_selection._split import KFold

        splitter = KFold(n_splits=cv)
    elif hasattr(cv, "split") and not isinstance(cv, (str, bytes)):
        splitter = cv
    else:
        raise TypeError(
            "cv must be an int or an object with a split method, "
            f"got {type(cv).__name__}."
        )

    if not isinstance(X, torch.Tensor):
        try:
            X = torch.as_tensor(X)
        except Exception as e:
            raise TypeError(f"X must be array-like. Got {type(X).__name__}.") from e
    if y is not None and not isinstance(y, torch.Tensor):
        try:
            y = torch.as_tensor(y)
        except Exception as e:
            raise TypeError(f"y must be array-like. Got {type(y).__name__}.") from e
    if y is not None and int(X.shape[0]) != int(y.shape[0]):
        raise ValueError(
            f"X and y have inconsistent lengths: {int(X.shape[0])} vs "
            f"{int(y.shape[0])}."
        )

    metric_fn = _resolve_scorer(scoring) if isinstance(scoring, str) else None

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(X, y):
        fold_est = clone(estimator)
        X_train, X_test = X[train_idx], X[test_idx]
        if y is None:
            fold_est.fit(X_train)
            fold_score = fold_est.score(X_test)
        else:
            y_train, y_test = y[train_idx], y[test_idx]
            fold_est.fit(X_train, y_train)
            if scoring is None:
                fold_score = fold_est.score(X_test, y_test)
            elif metric_fn is not None:
                y_pred = fold_est.predict(X_test)
                fold_score = metric_fn(y_test, y_pred)
            else:
                try:
                    fold_score = scoring(fold_est, X_test, y_test)
                except TypeError:
                    y_pred = fold_est.predict(X_test)
                    fold_score = scoring(y_test, y_pred)
        if isinstance(fold_score, torch.Tensor):
            fold_score = float(fold_score.item())
        scores.append(float(fold_score))

    return torch.tensor(scores, dtype=torch.float32)
