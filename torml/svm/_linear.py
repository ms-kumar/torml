"""Linear support vector machines with a PyTorch backend.

Both estimators minimize an L2-regularized empirical loss with the
Pegasos stochastic sub-gradient method.
"""

from __future__ import annotations

import torch

from torml.base import ClassifierMixin, RegressorMixin
from torml.utils._random import check_random_state
from torml.utils._validation import check_array, check_is_fitted, check_X_y


def _validate_C(C) -> float:
    """Validate the ``C`` regularization hyperparameter."""
    if isinstance(C, bool) or not isinstance(C, (int, float)):
        raise TypeError(f"C must be a float, got {type(C).__name__}.")
    if float(C) <= 0:
        raise ValueError(f"C must be > 0, got {C}.")
    return float(C)


def _validate_rate(name, value) -> int:
    """Validate non-negative int hyperparameters (epochs)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


class LinearSVC(ClassifierMixin):
    """Linear support vector classifier (hinge loss, Pegasos).

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength; smaller values mean stronger
        regularization.
    max_iter : int, default=1000
        Pegasos update steps (one sample each).
    tol : float, default=1e-3
        Stopping tolerance on the relative dual-gap proxy. ``None`` disables
        early stopping.
    random_state : int, torch.Generator or None, default=None
        Seed for sample shuffling.

    Attributes
    ----------
    classes_ : torch.Tensor of shape (2,)
        Sorted class labels (binary only).
    coef_ : torch.Tensor of shape (n_features,)
        Weight vector.
    intercept_ : torch.Tensor of shape (1,)
        Bias term.
    n_features_in_ : int
        Number of features seen during fit.
    n_iter_ : int
        Steps actually run.
    """

    name = "LinearSVC"

    def __init__(self, C=1.0, max_iter=1000, tol=1e-3, random_state=None):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the hyperplane with Pegasos updates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : LinearSVC
            Fitted classifier.
        """
        # pylint: disable=too-many-locals
        alpha = _validate_C(self.C)
        max_iter = _validate_rate("max_iter", self.max_iter)
        generator = check_random_state(self.random_state)
        X, y = check_X_y(X, y)
        flat = y.reshape(-1)
        classes, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        if int(classes.shape[0]) != 2:
            raise ValueError(
                "LinearSVC supports binary labels only, "
                f"got {int(classes.shape[0])} classes."
            )
        self.classes_ = classes
        signed = torch.where(
            inverse == 1,
            torch.tensor(1.0, dtype=X.dtype, device=X.device),
            torch.tensor(-1.0, dtype=X.dtype, device=X.device),
        )
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        self.n_features_in_ = n_features
        lam = 1.0 / (n_samples * alpha)
        w = torch.zeros(n_features, dtype=X.dtype, device=X.device)
        b = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        prev_obj: float | None = None
        n_iter = 0
        for t in range(1, max_iter + 1):
            i = int(
                torch.randint(
                    n_samples, (1,), generator=generator, device=X.device
                ).item()
            )
            eta = 1.0 / (lam * t)
            margin = signed[i] * (torch.dot(w, X[i]) + b)
            w = (1.0 - eta * lam) * w
            if margin < 1.0:
                w = w + eta * signed[i] * X[i]
                b = b + eta * signed[i] * 0.01
            n_iter = t
            if self.tol is not None and t % n_samples == 0:
                hinge = torch.clamp(1.0 - signed * (X @ w + b), min=0).mean()
                obj = 0.5 * lam * float(torch.dot(w, w)) + float(hinge)
                if prev_obj is not None and abs(prev_obj - obj) / max(
                    1.0, abs(prev_obj)
                ) < float(self.tol):
                    break
                prev_obj = obj
        self.coef_ = w
        self.intercept_ = b.reshape(1)
        self.n_iter_ = n_iter
        return self

    def decision_function(self, X):
        """Return signed distances to the hyperplane.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        scores : torch.Tensor of shape (n_samples,)
            Positive values predict ``classes_[1]``.
        """
        check_is_fitted(self, attributes=["coef_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but LinearSVC was fitted "
                f"with {int(self.n_features_in_)} features."
            )
        return Xt @ self.coef_ + self.intercept_.squeeze()

    def predict(self, X):
        """Predict class labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Predicted labels.
        """
        scores = self.decision_function(X)
        return torch.where(scores >= 0, self.classes_[1], self.classes_[0])


class LinearSVR(RegressorMixin):
    """Linear support vector regressor (epsilon-insensitive loss, Pegasos).

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength.
    epsilon : float, default=0.1
        Tube half-width with no penalty.
    max_iter : int, default=1000
        Pegasos update steps (one sample each).
    random_state : int, torch.Generator or None, default=None
        Seed for sample shuffling.

    Attributes
    ----------
    coef_ : torch.Tensor of shape (n_features,)
        Weight vector.
    intercept_ : torch.Tensor of shape (1,)
        Bias term.
    n_features_in_ : int
        Number of features seen during fit.
    n_iter_ : int
        Steps actually run.
    """

    name = "LinearSVR"

    def __init__(self, C=1.0, epsilon=0.1, max_iter=1000, random_state=None):
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the regression tube with Pegasos updates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        self : LinearSVR
            Fitted regressor.
        """
        alpha = _validate_C(self.C)
        max_iter = _validate_rate("max_iter", self.max_iter)
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)):
            raise TypeError(
                f"epsilon must be a float, got {type(self.epsilon).__name__}."
            )
        if float(self.epsilon) < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}.")
        generator = check_random_state(self.random_state)
        X, y = check_X_y(X, y)
        target = y.to(dtype=X.dtype, device=X.device).reshape(-1)
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        self.n_features_in_ = n_features
        lam = 1.0 / (n_samples * alpha)
        w = torch.zeros(n_features, dtype=X.dtype, device=X.device)
        b = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        for t in range(1, max_iter + 1):
            i = int(
                torch.randint(
                    n_samples, (1,), generator=generator, device=X.device
                ).item()
            )
            eta = 1.0 / (lam * t)
            residual = torch.dot(w, X[i]) + b - target[i]
            w = (1.0 - eta * lam) * w
            if residual > float(self.epsilon):
                w = w - eta * X[i]
                b = b - eta * 0.01
            elif residual < -float(self.epsilon):
                w = w + eta * X[i]
                b = b + eta * 0.01
        self.coef_ = w
        self.intercept_ = b.reshape(1)
        self.n_iter_ = max_iter
        return self

    def predict(self, X):
        """Predict targets for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, attributes=["coef_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but LinearSVR was fitted "
                f"with {int(self.n_features_in_)} features."
            )
        return Xt @ self.coef_ + self.intercept_.squeeze()
