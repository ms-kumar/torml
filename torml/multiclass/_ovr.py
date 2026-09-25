"""One-vs-rest multiclass strategy with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator, ClassifierMixin, clone
from torml.utils._validation import check_is_fitted


class OneVsRestClassifier(ClassifierMixin):
    """One-vs-rest wrapper around a binary estimator.

    Trains one clone per class (class vs rest with 0/1 labels) and
    predicts the class with the highest decision score.

    Parameters
    ----------
    estimator : estimator
        Binary estimator with ``decision_function`` or ``predict_proba``.

    Attributes
    ----------
    estimators_ : list of fitted estimators, one per class.
    classes_ : torch.Tensor or list
        Sorted unique labels.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "OneVsRestClassifier"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        """Fit one binary clone per class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Multiclass labels.

        Returns
        -------
        self : OneVsRestClassifier
            Fitted wrapper.
        """
        if not isinstance(self.estimator, BaseEstimator):
            raise TypeError(
                "estimator must be a BaseEstimator, "
                f"got {type(self.estimator).__name__}."
            )
        Xt = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
        flat = (
            torch.as_tensor(y).reshape(-1)
            if not isinstance(y, torch.Tensor)
            else y.reshape(-1)
        )
        if int(Xt.shape[0]) != int(flat.shape[0]):
            raise ValueError(
                f"X and y have inconsistent lengths: {int(Xt.shape[0])} "
                f"vs {int(flat.shape[0])}."
            )
        try:
            uniq = sorted(set(flat.tolist()))
        except TypeError as e:
            raise TypeError("Labels must be sortable.") from e
        if len(uniq) < 2:
            raise ValueError(f"Need at least 2 classes, got {len(uniq)}.")
        numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in uniq
        )
        self.classes_ = torch.as_tensor(uniq) if numeric else uniq
        self.estimators_ = []
        for c in uniq:
            cond = torch.as_tensor([v == c for v in flat.tolist()], device=Xt.device)
            binary = torch.where(
                cond,
                torch.tensor(1, device=Xt.device),
                torch.tensor(0, device=Xt.device),
            )
            self.estimators_.append(clone(self.estimator).fit(X, binary))
        first = self.estimators_[0]
        self.n_features_in_ = (
            int(first.n_features_in_)
            if hasattr(first, "n_features_in_")
            else int(Xt.shape[1])
        )
        return self

    def decision_function(self, X):
        """Return one score column per class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        scores : torch.Tensor of shape (n_samples, n_classes)
            Per-class decision scores.
        """
        check_is_fitted(self, attributes=["estimators_"])
        cols = []
        for est in self.estimators_:
            if hasattr(est, "decision_function"):
                s = torch.as_tensor(est.decision_function(X)).reshape(-1)
            elif hasattr(est, "predict_proba"):
                s = torch.as_tensor(est.predict_proba(X))[:, 1]
            else:
                raise AttributeError("Members need decision_function or predict_proba.")
            cols.append(s)
        return torch.stack(cols, dim=1)

    def predict(self, X):
        """Predict the highest-scoring class for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor or list of shape (n_samples,)
            Predicted labels.
        """
        idx = torch.argmax(self.decision_function(X), dim=1)
        if isinstance(self.classes_, torch.Tensor):
            return self.classes_[idx]
        return [self.classes_[int(i)] for i in idx.tolist()]
