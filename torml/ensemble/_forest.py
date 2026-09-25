"""Random forests with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import ClassifierMixin, RegressorMixin
from torml.ensemble._voting import _majority_vote
from torml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torml.utils._random import check_random_state
from torml.utils._validation import check_is_fitted


class RandomForestClassifier(ClassifierMixin):
    """Forest of decision trees on bootstrap subsamples.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees.
    max_depth : int or None, default=None
        Maximum depth per tree.
    random_state : int, torch.Generator or None, default=None
        Seed for bootstrap sampling.

    Attributes
    ----------
    estimators_ : list of fitted trees.
    classes_ : torch.Tensor
        Sorted union of tree classes.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "RandomForestClassifier"

    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        """Fit trees on bootstrap subsamples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : RandomForestClassifier
            Fitted forest.
        """
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
        yt = torch.as_tensor(y) if not isinstance(y, torch.Tensor) else y
        flat = yt.reshape(-1).to(Xt.device)
        n_samples = int(Xt.shape[0])
        try:
            uniq = sorted(set(flat.tolist()))
        except TypeError as e:
            raise TypeError("Labels must be sortable.") from e
        numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in uniq
        )
        self.classes_ = torch.as_tensor(uniq, device=Xt.device) if numeric else uniq
        self.estimators_ = []
        for _ in range(int(self.n_estimators)):
            idx = torch.randint(
                n_samples, (n_samples,), generator=generator, device=Xt.device
            )
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            self.estimators_.append(tree.fit(Xt[idx], flat[idx]))
        self.n_features_in_ = int(Xt.shape[1]) if Xt.ndim == 2 else 0
        return self

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


class RandomForestRegressor(RegressorMixin):
    """Forest of regression trees on bootstrap subsamples.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees.
    max_depth : int or None, default=None
        Maximum depth per tree.
    random_state : int, torch.Generator or None, default=None
        Seed for bootstrap sampling.

    Attributes
    ----------
    estimators_ : list of fitted trees.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "RandomForestRegressor"

    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        """Fit trees on bootstrap subsamples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        self : RandomForestRegressor
            Fitted forest.
        """
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
        _dtype = Xt.dtype if Xt.is_floating_point() else torch.float32
        yt = (
            torch.as_tensor(y, dtype=_dtype)
            if not isinstance(y, torch.Tensor)
            else y.to(dtype=_dtype)
        )
        flat = yt.reshape(-1).to(Xt.device)
        n_samples = int(Xt.shape[0])
        self.estimators_ = []
        for _ in range(int(self.n_estimators)):
            idx = torch.randint(
                n_samples, (n_samples,), generator=generator, device=Xt.device
            )
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            self.estimators_.append(tree.fit(Xt[idx], flat[idx]))
        self.n_features_in_ = int(Xt.shape[1]) if Xt.ndim == 2 else 0
        return self

    def predict(self, X):
        """Predict by averaging tree outputs.

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


__all__ = ["RandomForestClassifier", "RandomForestRegressor"]
