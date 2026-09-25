"""Gaussian Naive Bayes with a PyTorch backend."""

from __future__ import annotations

import math

import torch

from torml.base import ClassifierMixin
from torml.utils._validation import check_array, check_is_fitted


class GaussianNB(ClassifierMixin):
    """Gaussian Naive Bayes classifier.

    Assumes features follow a Gaussian distribution within each class and
    applies Bayes' theorem with a naive independence assumption.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest feature variance added to all variances
        for numerical stability. Must be >= 0.

    Attributes
    ----------
    classes_ : torch.Tensor of shape (n_classes,)
        Sorted unique class labels.
    class_count_ : torch.Tensor of shape (n_classes,)
        Samples per class.
    class_prior_ : torch.Tensor of shape (n_classes,)
        Empirical class probabilities.
    theta_ : torch.Tensor of shape (n_classes, n_features)
        Per-class feature means.
    var_ : torch.Tensor of shape (n_classes, n_features)
        Per-class feature variances (smoothed).
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "GaussianNB"

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def _validate_smoothing(self) -> float:
        vs = self.var_smoothing
        if isinstance(vs, bool) or not isinstance(vs, (int, float)):
            raise TypeError(f"var_smoothing must be a float, got {type(vs).__name__}.")
        vs = float(vs)
        if vs < 0:
            raise ValueError(f"var_smoothing must be >= 0, got {vs}.")
        return vs

    def fit(self, X, y):
        """Fit the model to ``X`` and ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : GaussianNB
            Fitted classifier.
        """
        vs = self._validate_smoothing()
        X = check_array(X, ensure_2d=True)
        if isinstance(y, torch.Tensor):
            flat = y.reshape(-1)
            n_y = int(flat.shape[0])
        elif isinstance(y, (list, tuple)):
            flat = list(y)
            n_y = len(flat)
        else:
            raise TypeError(
                "y must be a torch.Tensor, list or tuple, " f"got {type(y).__name__}."
            )
        if n_y != int(X.shape[0]):
            raise ValueError(
                f"X and y have inconsistent lengths: {int(X.shape[0])} vs {n_y}."
            )
        if isinstance(flat, torch.Tensor):
            labels = flat
        else:
            try:
                labels = torch.as_tensor(flat)
            except (TypeError, ValueError):
                labels = flat
        if isinstance(labels, torch.Tensor):
            self.classes_, inverse = torch.unique(
                labels, sorted=True, return_inverse=True
            )
            self.classes_ = self.classes_.to(device=X.device)
            inverse = inverse.to(dtype=torch.long, device=X.device)
        else:
            try:
                uniq = sorted(set(labels))
            except TypeError as e:
                raise TypeError("Labels must be sortable.") from e
            self.classes_ = uniq
            mapping = {c: i for i, c in enumerate(uniq)}
            inverse = torch.tensor(
                [mapping[v] for v in labels], dtype=torch.long, device=X.device
            )
        n_classes = (
            len(self.classes_)
            if isinstance(self.classes_, list)
            else int(self.classes_.shape[0])
        )
        n_features = int(X.shape[1])
        self.n_features_in_ = n_features
        self.n_samples_in_ = int(X.shape[0])

        self.class_count_ = torch.bincount(inverse, minlength=n_classes).to(
            dtype=X.dtype
        )
        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        self.theta_ = torch.empty(n_classes, n_features, dtype=X.dtype, device=X.device)
        self.var_ = torch.empty(n_classes, n_features, dtype=X.dtype, device=X.device)
        for c in range(n_classes):
            X_c = X[inverse == c]
            self.theta_[c] = X_c.mean(dim=0)
            self.var_[c] = X_c.var(dim=0, unbiased=False)
        epsilon = vs * X.var(dim=0, unbiased=False).max()
        self.var_ = self.var_ + epsilon
        self._epsilon = float(epsilon.item())
        return self

    def _joint_log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized log posteriors for ``X``."""
        check_is_fitted(self, attributes=["theta_", "var_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but GaussianNB was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        nll = -0.5 * torch.sum(torch.log(2.0 * math.pi * self.var_), dim=1)
        nll = nll + torch.log(self.class_prior_)
        nll = nll - 0.5 * torch.sum(
            ((Xt.unsqueeze(1) - self.theta_) ** 2) / self.var_, dim=2
        )
        return nll

    def predict_log_proba(self, X):
        """Return log probabilities for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        log_proba : torch.Tensor of shape (n_samples, n_classes)
            Log probabilities.
        """
        jll = self._joint_log_likelihood(X)
        return jll - torch.logsumexp(jll, dim=1, keepdim=True)

    def predict_proba(self, X):
        """Return class probabilities for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, n_classes)
            Per-class probabilities.
        """
        return torch.exp(self.predict_log_proba(X))

    def predict(self, X):
        """Predict class labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,) or list
            Predicted labels (tensor for tensor-fitted labels, else list).
        """
        jll = self._joint_log_likelihood(X)
        idx = torch.argmax(jll, dim=1)
        if isinstance(self.classes_, torch.Tensor):
            return self.classes_[idx]
        return [self.classes_[int(i)] for i in idx.tolist()]
