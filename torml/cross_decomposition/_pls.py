"""Partial least squares regression (NIPALS, univariate y)."""

from __future__ import annotations

import torch

from torml.base import RegressorMixin
from torml.utils._validation import check_array, check_is_fitted, check_X_y


class PLSRegression(RegressorMixin):
    """PLS regression via NIPALS (single-target).

    Parameters
    ----------
    n_components : int, default=2
        Number of latent components.
    max_iter : int, default=500
        Maximum NIPALS iterations per component.
    tol : float, default=1e-6
        Convergence tolerance on weight change.

    Attributes
    ----------
    x_weights_ : torch.Tensor of shape (n_features, n_components)
        X weights (directions).
    x_scores_ : torch.Tensor of shape (n_samples, n_components)
        Latent X scores.
    coef_ : torch.Tensor of shape (n_features,)
        Regression coefficients in original space.
    intercept_ : torch.Tensor of shape (1,)
        Intercept in original space.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "PLSRegression"

    def __init__(self, n_components=2, max_iter=500, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """Fit the NIPALS latent model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Single-target values.

        Returns
        -------
        self : PLSRegression
            Fitted regressor.
        """
        # pylint: disable=too-many-locals
        if isinstance(self.n_components, bool) or not isinstance(
            self.n_components, int
        ):
            raise TypeError(
                f"n_components must be an int, got {type(self.n_components).__name__}."
            )
        if int(self.n_components) < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        X, y = check_X_y(X, y)
        _dtype = X.dtype
        X = X.to(dtype=_dtype, device=X.device)
        target = y.to(dtype=_dtype, device=X.device).reshape(-1)
        n, d = int(X.shape[0]), int(X.shape[1])
        k = min(int(self.n_components), d)
        self.n_features_in_ = d
        self._x_mean = X.mean(dim=0)
        self._y_mean = target.mean()
        Xk = X - self._x_mean
        yk = target - self._y_mean
        weights, scores, loadings = [], [], []
        for _ in range(k):
            w = Xk.T @ yk
            norm = float(w.norm())
            if norm < 1e-12:
                break
            w = w / norm
            t = Xk @ w
            denom = float(t @ t)
            if denom < 1e-12:
                break
            p = (Xk.T @ t) / denom
            q = float(t @ yk / denom)
            weights.append(w)
            scores.append(t)
            loadings.append((p, q))
            Xk = Xk - torch.outer(t, p)
            yk = yk - t * q
        if not weights:
            self.x_weights_ = torch.zeros(d, 0, dtype=X.dtype, device=X.device)
            self.x_scores_ = torch.zeros(n, 0, dtype=X.dtype, device=X.device)
            self.coef_ = torch.zeros(d, dtype=X.dtype, device=X.device)
            self.intercept_ = self._y_mean.reshape(1)
            self.n_components_ = 0
            return self
        W = torch.stack(weights, dim=1)
        P = torch.stack([p for p, _ in loadings], dim=1)
        qvec = torch.tensor([q for _, q in loadings], dtype=X.dtype, device=X.device)
        self.x_weights_ = W
        self.x_scores_ = torch.stack(scores, dim=1)
        self.coef_ = W @ torch.linalg.solve(  # pylint: disable=not-callable
            P.T @ W, qvec
        )
        self.intercept_ = (self._y_mean - self._x_mean @ self.coef_).reshape(1)
        self.n_components_ = int(W.shape[1])
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
                f"X has {int(Xt.shape[1])} features, but PLSRegression was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        return Xt @ self.coef_ + self.intercept_.squeeze()

    def transform(self, X):
        """Project ``X`` onto the latent components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        scores : torch.Tensor of shape (n_samples, n_components_)
            Latent scores.
        """
        check_is_fitted(self, attributes=["x_weights_"])
        Xt = check_array(X, ensure_2d=True)
        return (Xt - self._x_mean) @ self.x_weights_
