"""Linear discriminant analysis with a PyTorch backend (SVD solver)."""

from __future__ import annotations

import torch

from torml.base import ClassifierMixin, TransformerMixin
from torml.utils._validation import check_array, check_is_fitted, check_X_y


class LinearDiscriminantAnalysis(ClassifierMixin, TransformerMixin):
    """LDA classifier with optional dimensionality reduction.

    Parameters
    ----------
    n_components : int or None, default=None
        Discriminant dimensions to keep (at most n_classes - 1).

    Attributes
    ----------
    classes_ : torch.Tensor of shape (n_classes,)
        Sorted unique labels.
    priors_ : torch.Tensor of shape (n_classes,)
        Empirical class priors.
    means_ : torch.Tensor of shape (n_classes, n_features)
        Per-class means.
    coef_ : torch.Tensor of shape (n_features,) or (n_features, n_classes)
        Log-posterior coefficients (binary vector, else per-class matrix).
    intercept_ : torch.Tensor
        Log-posterior intercepts.
    scalings_ : torch.Tensor of shape (n_features, n_discriminants)
        Whitening directions for :meth:`transform`.
    explained_variance_ratio_ : torch.Tensor
        Discriminant variance fractions.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "LinearDiscriminantAnalysis"

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        """Estimate class statistics and discriminant directions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : LinearDiscriminantAnalysis
            Fitted model.
        """
        # pylint: disable=too-many-locals
        X, y = check_X_y(X, y)
        X = X.to(dtype=torch.float32)
        flat = y.reshape(-1)
        self.classes_, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        inverse = inverse.to(dtype=torch.long)
        n, d = int(X.shape[0]), int(X.shape[1])
        n_classes = int(self.classes_.shape[0])
        if n_classes < 2:
            raise ValueError(f"Need at least 2 classes, got {n_classes}.")
        self.n_features_in_ = d
        counts = torch.bincount(inverse, minlength=n_classes).to(dtype=torch.float32)
        self.priors_ = counts / counts.sum()
        self.means_ = torch.stack(
            [X[inverse == c].mean(dim=0) for c in range(n_classes)]
        )
        centered = X - self.means_[inverse]
        _, singular, Vt = torch.linalg.svd(  # pylint: disable=not-callable
            centered, full_matrices=False
        )
        rank = int((singular > singular[0] * max(n, d) * 1e-12).sum())
        V, S = Vt.T[:, :rank], singular[:rank]
        X_orth = (centered @ V) / S.clamp(min=1e-12)
        grand = X_orth.mean(dim=0)
        between = torch.zeros(rank, rank, dtype=torch.float32)
        for c in range(n_classes):
            diff = X_orth[inverse == c].mean(dim=0) - grand
            between = between + counts[c] * torch.outer(diff, diff)
        between = between / (n - n_classes)
        evals, evecs = torch.linalg.eigh(between)  # pylint: disable=not-callable
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]
        max_comp = min(n_classes - 1, rank)
        if self.n_components is None:
            k = max_comp
        elif isinstance(self.n_components, bool) or not isinstance(
            self.n_components, int
        ):
            raise TypeError(
                "n_components must be int or None, "
                f"got {type(self.n_components).__name__}."
            )
        elif not 1 <= int(self.n_components) <= max_comp:
            raise ValueError(
                f"n_components must satisfy 1 <= n_components <= {max_comp}, "
                f"got {self.n_components}."
            )
        else:
            k = int(self.n_components)
        scalings = (V / S.clamp(min=1e-12)) @ evecs[:, :k]
        self.scalings_ = scalings
        total = float(evals[:max_comp].sum())
        self.explained_variance_ratio_ = (
            evals[:k] / total if total > 0 else torch.zeros(k)
        )
        self._scalings_full = (V / S.clamp(min=1e-12)) @ evecs[:, :max_comp]
        proj_means = self.means_ @ self._scalings_full
        if n_classes == 2:
            direction = (proj_means[1] - proj_means[0]) / 1.0
            w = self._scalings_full @ direction
            mid = (proj_means[0] + proj_means[1]) / 2
            b = float(torch.log(self.priors_[1] / self.priors_[0])) - float(
                mid @ direction
            )
            self.coef_ = w
            self.intercept_ = torch.tensor([b])
        else:
            coefs, intercepts = [], []
            for c in range(n_classes):
                w = self._scalings_full @ proj_means[c]
                b = float(torch.log(self.priors_[c])) - 0.5 * float(
                    proj_means[c] @ proj_means[c]
                )
                coefs.append(w)
                intercepts.append(b)
            self.coef_ = torch.stack(coefs, dim=1)
            self.intercept_ = torch.tensor(intercepts)
        return self

    def _scores(self, X):
        """Unnormalized log posteriors for ``X``."""
        check_is_fitted(self, attributes=["coef_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but LDA was fitted "
                f"with {int(self.n_features_in_)} features."
            )
        if self.coef_.ndim == 1:
            return Xt @ self.coef_ + self.intercept_.squeeze()
        return Xt @ self.coef_ + self.intercept_

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
        scores = self._scores(X)
        if scores.ndim == 1:
            idx = (scores >= 0).long()
        else:
            idx = torch.argmax(scores, dim=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        """Return class probabilities for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, n_classes)
            Softmaxed log posteriors.
        """
        scores = self._scores(X)
        if scores.ndim == 1:
            proba_1 = torch.sigmoid(scores)
            return torch.stack([1 - proba_1, proba_1], dim=1)
        return torch.softmax(scores, dim=1)

    def transform(self, X):
        """Project ``X`` onto discriminant directions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        Xt : torch.Tensor of shape (n_samples, n_discriminants)
            Projected data.
        """
        check_is_fitted(self, attributes=["scalings_"])
        Xt = check_array(X, ensure_2d=True, dtype=torch.float32)
        return Xt @ self.scalings_

    def _transform(self, X):
        return self.transform(X)
