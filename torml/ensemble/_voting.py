"""Voting ensembles with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone


def _validate_estimators(estimators) -> list:
    """Validate the ``estimators`` list of (name, estimator) pairs."""
    if not isinstance(estimators, (list, tuple)) or len(estimators) == 0:
        raise TypeError("estimators must be a non-empty list of (name, estimator).")
    cleaned = []
    seen = set()
    for item in estimators:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise TypeError("Each entry must be a (name, estimator) pair.")
        name, est = item
        if not isinstance(name, str):
            raise TypeError(f"Estimator name must be a str, got {type(name).__name__}.")
        if name in seen:
            raise ValueError(f"Duplicate estimator name {name!r}.")
        if name == "drop":
            continue
        if not isinstance(est, BaseEstimator):
            raise TypeError(
                f"Estimator {name!r} must be a BaseEstimator, "
                f"got {type(est).__name__}."
            )
        seen.add(name)
        cleaned.append((name, est))
    if not cleaned:
        raise ValueError("All estimators are 'drop'; need at least one.")
    return cleaned


class VotingClassifier(ClassifierMixin):
    """Majority/soft vote over classifiers.

    Parameters
    ----------
    estimators : list of (str, estimator)
        Classifiers to combine. ``'drop'`` entries are skipped.
    voting : {'hard', 'soft'}, default='hard'
        ``'hard'`` predicts the majority label; ``'soft'`` averages
        ``predict_proba`` outputs (all members must implement it).

    Attributes
    ----------
    estimators_ : list of (str, estimator)
        Fitted member clones.
    classes_ : torch.Tensor
        Sorted union of member classes.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "VotingClassifier"

    def __init__(self, estimators, voting="hard"):
        self.estimators = estimators
        self.voting = voting

    def get_params(self, deep=True):
        """Get parameters including per-member ``name__param`` entries."""
        params = {"estimators": self.estimators, "voting": self.voting}
        if deep:
            for name, est in _validate_estimators(self.estimators):
                for k, v in est.get_params(deep=True).items():
                    params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params):
        """Set parameters including per-member ``name__param`` entries."""
        nested: dict = {}
        for key, value in params.items():
            if "__" in key:
                name, sub = key.split("__", 1)
                nested.setdefault(name, {})[sub] = value
            elif key == "estimators":
                self.estimators = value
            elif key == "voting":
                self.voting = value
            else:
                raise ValueError(f"Invalid parameter {key!r} for VotingClassifier.")
        if nested:
            rebuilt = []
            members = dict(_validate_estimators(self.estimators))
            for name, est in members.items():
                if name in nested:
                    est = clone(est).set_params(**nested[name])
                rebuilt.append((name, est))
            self.estimators = rebuilt
        return self

    def fit(self, X, y):
        """Fit all members on ``X``, ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : VotingClassifier
            Fitted ensemble.
        """
        if self.voting not in ("hard", "soft"):
            raise ValueError(f"voting must be 'hard' or 'soft', got {self.voting!r}.")
        members = _validate_estimators(self.estimators)
        if self.voting == "soft":
            for name, est in members:
                if not hasattr(est, "predict_proba"):
                    raise TypeError(
                        f"Estimator {name!r} needs predict_proba for soft voting."
                    )
        flat = (
            torch.as_tensor(y).reshape(-1)
            if not isinstance(y, torch.Tensor)
            else y.reshape(-1)
        )
        try:
            uniq = sorted(set(flat.tolist()))
        except TypeError as e:
            raise TypeError("Labels must be sortable.") from e
        numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in uniq
        )
        self.classes_ = torch.as_tensor(uniq) if numeric else uniq
        self.estimators_ = [(name, clone(est).fit(X, y)) for name, est in members]
        first = self.estimators_[0][1]
        if hasattr(first, "n_features_in_"):
            self.n_features_in_ = int(first.n_features_in_)
        else:
            arr = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
            self.n_features_in_ = int(arr.shape[1]) if arr.ndim == 2 else 0
        return self

    def _align_proba(self, proba, member_classes):
        """Map a member proba matrix onto ``self.classes_`` columns."""
        if isinstance(self.classes_, torch.Tensor) and isinstance(
            member_classes, torch.Tensor
        ):
            if torch.equal(self.classes_, member_classes):
                return proba
        index = []
        own = (
            self.classes_.tolist()
            if isinstance(self.classes_, torch.Tensor)
            else list(self.classes_)
        )
        other = (
            member_classes.tolist()
            if isinstance(member_classes, torch.Tensor)
            else list(member_classes)
        )
        pos = {c: i for i, c in enumerate(other)}
        for c in own:
            index.append(pos[c])
        return proba[:, torch.tensor(index, dtype=torch.long)]

    def predict_proba(self, X):
        """Average member probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, n_classes)
            Averaged probabilities.
        """
        from torml.utils._validation import check_is_fitted

        check_is_fitted(self, attributes=["estimators_"])
        if self.voting != "soft":
            raise AttributeError("predict_proba requires voting='soft'.")
        total = None
        for _, est in self.estimators_:
            aligned = self._align_proba(est.predict_proba(X), est.classes_)
            total = aligned if total is None else total + aligned
        return total / len(self.estimators_)

    def predict(self, X):
        """Predict labels by vote.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor or list of shape (n_samples,)
            Voted labels.
        """
        from torml.utils._validation import check_is_fitted

        check_is_fitted(self, attributes=["estimators_"])
        if self.voting == "soft":
            idx = torch.argmax(self.predict_proba(X), dim=1)
        else:
            own = (
                self.classes_.tolist()
                if isinstance(self.classes_, torch.Tensor)
                else list(self.classes_)
            )
            pos = {c: i for i, c in enumerate(own)}
            cols = []
            for _, est in self.estimators_:
                pred = est.predict(X)
                vals = pred.tolist() if isinstance(pred, torch.Tensor) else list(pred)
                cols.append(
                    torch.tensor(
                        [
                            pos[v.item() if isinstance(v, torch.Tensor) else v]
                            for v in vals
                        ],
                        dtype=torch.long,
                    )
                )
            idx = torch.mode(torch.stack(cols), dim=0).values
        if isinstance(self.classes_, torch.Tensor):
            return self.classes_[idx]
        return [self.classes_[int(i)] for i in idx.tolist()]


class VotingRegressor(RegressorMixin):
    """Average over regressors.

    Parameters
    ----------
    estimators : list of (str, estimator)
        Regressors to combine. ``'drop'`` entries are skipped.

    Attributes
    ----------
    estimators_ : list of (str, estimator)
        Fitted member clones.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "VotingRegressor"

    def __init__(self, estimators):
        self.estimators = estimators

    def get_params(self, deep=True):
        """Get parameters including per-member ``name__param`` entries."""
        params = {"estimators": self.estimators}
        if deep:
            for name, est in _validate_estimators(self.estimators):
                for k, v in est.get_params(deep=True).items():
                    params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params):
        """Set parameters including per-member ``name__param`` entries."""
        nested: dict = {}
        for key, value in params.items():
            if "__" in key:
                name, sub = key.split("__", 1)
                nested.setdefault(name, {})[sub] = value
            elif key == "estimators":
                self.estimators = value
            else:
                raise ValueError(f"Invalid parameter {key!r} for VotingRegressor.")
        if nested:
            members = dict(_validate_estimators(self.estimators))
            rebuilt = []
            for name, est in members.items():
                if name in nested:
                    est = clone(est).set_params(**nested[name])
                rebuilt.append((name, est))
            self.estimators = rebuilt
        return self

    def fit(self, X, y):
        """Fit all members on ``X``, ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        self : VotingRegressor
            Fitted ensemble.
        """
        members = _validate_estimators(self.estimators)
        self.estimators_ = [(name, clone(est).fit(X, y)) for name, est in members]
        first = self.estimators_[0][1]
        self.n_features_in_ = (
            int(first.n_features_in_)
            if hasattr(first, "n_features_in_")
            else int(torch.as_tensor(X).shape[1])
        )
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
        from torml.utils._validation import check_is_fitted

        check_is_fitted(self, attributes=["estimators_"])
        stacked = torch.stack(
            [
                torch.as_tensor(est.predict(X), dtype=torch.float32)
                for _, est in self.estimators_
            ]
        )
        return stacked.mean(dim=0)
