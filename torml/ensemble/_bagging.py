"""Bagging ensembles with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from torml.ensemble._voting import _majority_vote
from torml.utils._random import check_random_state
from torml.utils._validation import check_is_fitted


def _resolve_n_samples(max_samples, n_samples: int) -> int:
    """Convert ``max_samples`` (int/float) to an absolute count."""
    if isinstance(max_samples, bool):
        raise TypeError("max_samples must be int or float.")
    if isinstance(max_samples, int):
        if not 1 <= max_samples <= n_samples:
            raise ValueError(
                f"max_samples={max_samples} must satisfy 1 <= max_samples "
                f"<= n_samples ({n_samples})."
            )
        return max_samples
    if isinstance(max_samples, float):
        if not 0.0 < max_samples <= 1.0:
            raise ValueError(f"max_samples={max_samples} must be in (0, 1].")
        return max(1, int(round(max_samples * n_samples)))
    raise TypeError(
        f"max_samples must be int or float, got {type(max_samples).__name__}."
    )


class _BaseBagging(BaseEstimator):
    """Shared bootstrap fitting for bagging ensembles."""

    # pylint: disable=no-member

    def _fit_members(self, X, y, bootstrap: bool):
        """Fit ``n_estimators`` clones on bootstrap subsamples."""
        if isinstance(self.n_estimators, bool) or not isinstance(
            self.n_estimators, int
        ):
            raise TypeError(
                f"n_estimators must be an int, got {type(self.n_estimators).__name__}."
            )
        if int(self.n_estimators) < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        device = X.device if isinstance(X, torch.Tensor) else None
        generator = check_random_state(self.random_state, device)
        Xt = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
        n_samples = int(Xt.shape[0])
        k = _resolve_n_samples(self.max_samples, n_samples)
        members = []
        for _ in range(int(self.n_estimators)):
            if bootstrap:
                idx = torch.randint(
                    n_samples, (k,), generator=generator, device=Xt.device
                )
            else:
                idx = torch.randperm(n_samples, generator=generator, device=Xt.device)[
                    :k
                ]
            members.append((Xt, idx))
        return members, generator


class BaggingClassifier(ClassifierMixin, _BaseBagging):
    """Bagging classifier over bootstrap fits.

    Parameters
    ----------
    estimator : estimator or None, default=None
        Base estimator (default: ``DecisionTreeClassifier``). Must support
        ``fit``/``predict``/``predict_proba``.
    n_estimators : int, default=10
        Number of members.
    max_samples : int or float, default=1.0
        Samples per member (count, or fraction when float).
    bootstrap : bool, default=True
        Sample with replacement when True.
    random_state : int, torch.Generator or None, default=None
        Seed for bootstrap sampling.

    Attributes
    ----------
    estimators_ : list of fitted estimators.
    classes_ : torch.Tensor
        Sorted union of member classes.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "BaggingClassifier"

    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        max_samples=1.0,
        bootstrap=True,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _base(self):
        """Return the configured base estimator."""
        if self.estimator is None:
            from torml.tree import DecisionTreeClassifier

            return DecisionTreeClassifier()
        return self.estimator

    def fit(self, X, y):
        """Fit members on bootstrap subsamples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : BaggingClassifier
            Fitted ensemble.
        """
        base = self._base()
        members, _ = self._fit_members(X, y, bool(self.bootstrap))
        Xt = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
        yt = torch.as_tensor(y) if not isinstance(y, torch.Tensor) else y
        flat = yt.reshape(-1).to(Xt.device)
        try:
            uniq = sorted(set(flat.tolist()))
        except TypeError as e:
            raise TypeError("Labels must be sortable.") from e
        numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in uniq
        )
        self.classes_ = torch.as_tensor(uniq, device=Xt.device) if numeric else uniq
        self.estimators_ = [clone(base).fit(Xt[idx], flat[idx]) for _, idx in members]
        self.n_features_in_ = int(Xt.shape[1]) if Xt.ndim == 2 else 0
        return self

    def predict_proba(self, X):
        """Average member probabilities for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, n_classes)
            Averaged probabilities.
        """
        check_is_fitted(self, attributes=["estimators_"])
        own = (
            self.classes_.tolist()
            if isinstance(self.classes_, torch.Tensor)
            else list(self.classes_)
        )
        pos = {c: i for i, c in enumerate(own)}
        total = None
        for est in self.estimators_:
            proba = est.predict_proba(X)
            other = (
                est.classes_.tolist()
                if isinstance(est.classes_, torch.Tensor)
                else list(est.classes_)
            )
            aligned = proba[
                :,
                torch.tensor(
                    [pos[c] for c in other],
                    dtype=torch.long,
                    device=proba.device if isinstance(proba, torch.Tensor) else None,
                ),
            ]
            total = aligned if total is None else total + aligned
        return total / len(self.estimators_)

    def predict(self, X):
        """Predict labels by majority vote.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor or list of shape (n_samples,)
            Voted labels.
        """
        check_is_fitted(self, attributes=["estimators_"])
        own = (
            self.classes_.tolist()
            if isinstance(self.classes_, torch.Tensor)
            else list(self.classes_)
        )
        pos = {c: i for i, c in enumerate(own)}
        cols = []
        device = (
            self.classes_.device
            if isinstance(self.classes_, torch.Tensor)
            else (X.device if isinstance(X, torch.Tensor) else None)
        )
        for est in self.estimators_:
            pred = est.predict(X)
            vals = pred.tolist() if isinstance(pred, torch.Tensor) else list(pred)
            cols.append(
                torch.tensor(
                    [pos[v.item() if isinstance(v, torch.Tensor) else v] for v in vals],
                    dtype=torch.long,
                    device=device,
                )
            )
            idx = _majority_vote(torch.stack(cols))
        if isinstance(self.classes_, torch.Tensor):
            return self.classes_[idx]
        return [self.classes_[int(i)] for i in idx.tolist()]


class BaggingRegressor(RegressorMixin, _BaseBagging):
    """Bagging regressor over bootstrap fits.

    Parameters
    ----------
    estimator : estimator or None, default=None
        Base estimator (default: ``DecisionTreeRegressor``).
    n_estimators : int, default=10
        Number of members.
    max_samples : int or float, default=1.0
        Samples per member (count, or fraction when float).
    bootstrap : bool, default=True
        Sample with replacement when True.
    random_state : int, torch.Generator or None, default=None
        Seed for bootstrap sampling.

    Attributes
    ----------
    estimators_ : list of fitted estimators.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "BaggingRegressor"

    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        max_samples=1.0,
        bootstrap=True,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _base(self):
        """Return the configured base estimator."""
        if self.estimator is None:
            from torml.tree import DecisionTreeRegressor

            return DecisionTreeRegressor()
        return self.estimator

    def fit(self, X, y):
        """Fit members on bootstrap subsamples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        self : BaggingRegressor
            Fitted ensemble.
        """
        base = self._base()
        members, _ = self._fit_members(X, y, bool(self.bootstrap))
        Xt = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
        _dtype = Xt.dtype if Xt.is_floating_point() else torch.float32
        yt = (
            torch.as_tensor(y, dtype=_dtype)
            if not isinstance(y, torch.Tensor)
            else y.to(dtype=_dtype)
        )
        flat = yt.reshape(-1).to(Xt.device)
        self.estimators_ = [clone(base).fit(Xt[idx], flat[idx]) for _, idx in members]
        self.n_features_in_ = int(Xt.shape[1]) if Xt.ndim == 2 else 0
        return self

    def predict(self, X):
        """Predict by averaging member outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Averaged predictions.
        """
        check_is_fitted(self, attributes=["estimators_"])
        stacked = torch.stack(
            [torch.as_tensor(est.predict(X)) for est in self.estimators_]
        )
        return stacked.mean(dim=0)
