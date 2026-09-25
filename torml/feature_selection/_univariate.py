"""Univariate feature selection with a PyTorch backend."""

from __future__ import annotations

import math

import torch

from torml.base import TransformerMixin
from torml.utils._validation import check_array, check_is_fitted, check_X_y


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    max_iter, eps, tiny = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _f_pvalue(score: float, df_between: int, df_within: int) -> float:
    """Survival probability P(F > score) for F(df_between, df_within)."""
    d1, d2 = float(df_between), float(max(df_within, 1))
    z = d2 / (d2 + d1 * max(score, 0.0))
    return _betai(d2 / 2.0, d1 / 2.0, min(max(z, 0.0), 1.0))


def _group_scatter(Xt, flat, classes, overall):
    """Between/within sum of squares per feature for class groups."""
    d = int(Xt.shape[1])
    ss_between = torch.zeros(d, dtype=Xt.dtype, device=Xt.device)
    ss_within = torch.zeros(d, dtype=Xt.dtype, device=Xt.device)
    for c in classes.tolist():
        group = Xt[flat == c]
        n_c = int(group.shape[0])
        diff = group.mean(dim=0) - overall
        ss_between = ss_between + n_c * diff * diff
        ss_within = ss_within + ((group - group.mean(dim=0)) ** 2).sum(dim=0)
    return ss_between, ss_within


def f_classif(X, y):
    """Compute ANOVA F-scores for classification.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data.
    y : array-like of shape (n_samples,)
        Class labels.

    Returns
    -------
    scores : torch.Tensor of shape (n_features,)
        F-statistics per feature.
    pvalues : torch.Tensor of shape (n_features,)
        Survival probabilities under the F distribution.
    """
    Xt, yt = check_X_y(X, y)
    Xt = Xt.to(dtype=Xt.dtype)
    flat = yt.reshape(-1)
    classes = torch.unique(flat, sorted=True)
    n = int(Xt.shape[0])
    n_classes = int(classes.shape[0])
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {n_classes}.")
    overall = Xt.mean(dim=0)
    ss_between, ss_within = _group_scatter(Xt, flat, classes, overall)
    df_between, df_within = n_classes - 1, n - n_classes
    ms_between = ss_between / df_between
    denom = max(df_within, 1)
    ms_within = ss_within / denom if df_within > 0 else torch.zeros_like(ss_within)
    with torch.no_grad():
        scores = torch.where(
            ms_within > 0,
            ms_between / ms_within.clamp(min=1e-12),
            torch.zeros_like(ms_between),
        )
        # P(X > x) for F(df_between, df_within) via the incomplete beta.
        pvalues = torch.tensor(
            [_f_pvalue(float(s), df_between, df_within) for s in scores.tolist()],
            dtype=Xt.dtype,
            device=Xt.device,
        )
    return scores, pvalues


class SelectKBest(TransformerMixin):
    """Select the k highest-scoring features.

    Parameters
    ----------
    score_func : callable, default=f_classif
        ``(X, y) -> (scores, pvalues)``.
    k : int or 'all', default=10
        Features to keep.

    Attributes
    ----------
    scores_ : torch.Tensor of shape (n_features,)
        Scores per feature.
    pvalues_ : torch.Tensor of shape (n_features,)
        P-values per feature.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "SelectKBest"

    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        """Score features on ``X``, ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : SelectKBest
            Fitted selector.
        """
        if not callable(self.score_func):
            raise TypeError("score_func must be callable.")
        Xt, yt = check_X_y(X, y)
        n_features = int(Xt.shape[1])
        self.n_features_in_ = n_features
        if isinstance(self.k, str):
            if self.k != "all":
                raise ValueError(f"k must be int or 'all', got {self.k!r}.")
            k = n_features
        elif isinstance(self.k, bool) or not isinstance(self.k, int):
            raise TypeError(f"k must be int or 'all', got {type(self.k).__name__}.")
        elif not 1 <= int(self.k) <= n_features:
            raise ValueError(
                f"k={self.k} must satisfy 1 <= k <= n_features ({n_features})."
            )
        else:
            k = int(self.k)
        scores, pvalues = self.score_func(Xt, yt)
        scores = torch.as_tensor(scores, dtype=Xt.dtype, device=Xt.device).reshape(-1)
        pvalues = torch.as_tensor(pvalues, dtype=Xt.dtype, device=Xt.device).reshape(-1)
        if int(scores.shape[0]) != n_features:
            raise ValueError("score_func returned wrong number of scores.")
        self.scores_ = scores
        self.pvalues_ = pvalues
        order = torch.argsort(
            torch.nan_to_num(scores, nan=float("-inf")), descending=True, stable=True
        )
        mask = torch.zeros(n_features, dtype=torch.bool, device=Xt.device)
        mask[order[:k]] = True
        self._mask = mask
        return self

    def fit_transform(self, X, y):
        """Score and select columns of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, k)
            Selected columns.
        """
        return self.fit(X, y).transform(X)

    def transform(self, X):
        """Keep the selected columns of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, k)
            Selected columns.
        """
        check_is_fitted(self, attributes=["_mask"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but SelectKBest was "
                f"fitted with {int(self.n_features_in_)} features."
            )
        return Xt[:, self._mask]

    def _transform(self, X):
        return self.transform(X)

    def get_support(self, indices=False):
        """Return the selection mask (or indices).

        Parameters
        ----------
        indices : bool, default=False
            Return integer indices instead of a mask.

        Returns
        -------
        support : torch.Tensor
            Boolean mask or selected indices.
        """
        check_is_fitted(self, attributes=["_mask"])
        if indices:
            return torch.where(self._mask)[0]
        return self._mask
