"""Label propagation with a PyTorch backend (kNN graph, hard clamping)."""

from __future__ import annotations

import torch

from torml.base import ClassifierMixin
from torml.utils._validation import check_array, check_is_fitted


class LabelPropagation(ClassifierMixin):
    """Propagate labels from labeled to unlabeled points (-1).

    Builds a symmetric kNN graph, row-normalizes it, and iterates the
    label distributions while clamping the labeled rows.

    Parameters
    ----------
    n_neighbors : int, default=7
        Graph connectivity.
    max_iter : int, default=1000
        Maximum propagation iterations.
    tol : float, default=1e-3
        Convergence tolerance on distribution change.

    Attributes
    ----------
    classes_ : torch.Tensor
        Sorted unique labeled classes.
    label_distributions_ : torch.Tensor of shape (n_samples, n_classes)
        Final soft assignments.
    transduction_ : torch.Tensor of shape (n_samples,)
        Hard labels for the fitted samples.
    n_features_in_ : int
        Number of features seen during fit.
    n_iter_ : int
        Iterations run.
    """

    name = "LabelPropagation"

    def __init__(self, n_neighbors=7, max_iter=1000, tol=1e-3):
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """Propagate ``y`` labels (with -1 unlabeled) over ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            All samples (labeled and unlabeled).
        y : array-like of shape (n_samples,)
            Labels with -1 for unlabeled points.

        Returns
        -------
        self : LabelPropagation
            Fitted propagator.
        """
        # pylint: disable=too-many-locals
        if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, int):
            raise TypeError(
                f"n_neighbors must be an int, got {type(self.n_neighbors).__name__}."
            )
        if int(self.n_neighbors) < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors}.")
        Xt = check_array(X, ensure_2d=True)
        flat = (
            torch.as_tensor(y).reshape(-1)
            if not isinstance(y, torch.Tensor)
            else y.reshape(-1)
        )
        n = int(Xt.shape[0])
        if int(flat.shape[0]) != n:
            raise ValueError(
                f"X and y have inconsistent lengths: {n} vs {int(flat.shape[0])}."
            )
        self.n_features_in_ = int(Xt.shape[1])
        labeled_mask = flat != -1
        if not bool(labeled_mask.any()):
            raise ValueError("At least one labeled sample (y != -1) is required.")
        labeled = flat[labeled_mask].tolist()
        try:
            uniq = sorted(set(labeled))
        except TypeError as e:
            raise TypeError("Labels must be sortable.") from e
        self.classes_ = torch.as_tensor(uniq)
        n_classes = len(uniq)
        pos = {c: i for i, c in enumerate(uniq)}

        dist = torch.cdist(Xt, Xt, p=2)
        dist.fill_diagonal_(float("inf"))
        k = min(int(self.n_neighbors), n - 1)
        knn = torch.topk(dist, k=k, dim=1, largest=False).indices
        weight = torch.zeros(n, n, dtype=Xt.dtype, device=Xt.device)
        sigma = float(torch.median(dist[dist < float("inf")])) + 1e-12
        for i in range(n):
            weight[i, knn[i]] = torch.exp(-dist[i, knn[i]] ** 2 / (2 * sigma**2))
        weight = torch.maximum(weight, weight.T)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-12)

        Y = torch.zeros(n, n_classes, dtype=Xt.dtype, device=Xt.device)
        for i in torch.where(labeled_mask)[0].tolist():
            v = flat[i].item()
            Y[i, pos[v]] = 1.0
        F = Y.clone()
        for it in range(1, int(self.max_iter) + 1):
            F_new = weight @ F
            F_new[labeled_mask] = Y[labeled_mask]
            delta = float(torch.abs(F_new - F).sum())
            F = F_new
            if delta < float(self.tol):
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = int(self.max_iter)
        self.label_distributions_ = F
        self.transduction_ = self.classes_[torch.argmax(F, dim=1)]
        return self

    def predict(self, X):
        """Predict labels for points seen during fit, by index alignment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must be the fitted samples in order (transductive).

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Propagated labels.
        """
        check_is_fitted(self, attributes=["transduction_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[0]) != int(self.transduction_.shape[0]):
            raise ValueError(
                "LabelPropagation is transductive: predict on the fitted X."
            )
        return self.transduction_

    def predict_proba(self, X):
        """Return label distributions for the fitted samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must be the fitted samples in order (transductive).

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, n_classes)
            Soft assignments.
        """
        check_is_fitted(self, attributes=["label_distributions_"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[0]) != int(self.label_distributions_.shape[0]):
            raise ValueError(
                "LabelPropagation is transductive: predict on the fitted X."
            )
        return self.label_distributions_
