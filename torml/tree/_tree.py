"""CART decision trees with a PyTorch backend."""

from __future__ import annotations

import torch

from torml.base import ClassifierMixin, RegressorMixin
from torml.utils._validation import check_array, check_is_fitted


def _gini(counts: torch.Tensor) -> torch.Tensor:
    """Gini impurity for count vectors (..., n_classes)."""
    total = counts.sum(dim=-1, keepdim=True).clamp(min=1)
    prob = counts / total
    return 1.0 - (prob * prob).sum(dim=-1)


def _entropy(counts: torch.Tensor) -> torch.Tensor:
    """Entropy impurity for count vectors (..., n_classes)."""
    total = counts.sum(dim=-1, keepdim=True).clamp(min=1)
    prob = counts / total
    return -(prob * torch.log(prob.clamp(min=1e-12))).sum(dim=-1)


def _best_split_classifier(X, y_idx, n_classes, criterion):
    """Find the best (feature, threshold, gain) split for labeled rows."""
    # pylint: disable=too-many-locals
    n, n_features = int(X.shape[0]), int(X.shape[1])
    impurity_fn = _gini if criterion == "gini" else _entropy
    total_counts = torch.bincount(y_idx, minlength=n_classes).to(dtype=X.dtype)
    parent = impurity_fn(total_counts)
    best = (0.0, -1, 0.0)
    for f in range(n_features):
        col = X[:, f]
        order = torch.argsort(col, stable=True)
        sorted_x = col[order]
        valid = sorted_x[:-1] != sorted_x[1:]
        if not bool(valid.any()):
            continue
        one_hot = torch.zeros(n, n_classes, dtype=X.dtype, device=X.device)
        one_hot[torch.arange(n, device=X.device), y_idx[order]] = 1.0
        left_counts = torch.cumsum(one_hot, dim=0)[:-1]
        left_n = torch.arange(1, n, dtype=X.dtype, device=X.device)
        right_counts = total_counts - left_counts
        right_n = float(n) - left_n
        gain = (
            float(n) * parent
            - left_n * impurity_fn(left_counts)
            - right_n * impurity_fn(right_counts)
        )
        gain = torch.where(valid, gain, torch.zeros_like(gain))
        i = int(torch.argmax(gain).item())
        g = float(gain[i].item())
        if g > best[0]:
            best = (g, f, float((sorted_x[i] + sorted_x[i + 1]).item()) / 2.0)
    return best


def _best_split_regressor(X, y):
    """Find the best (feature, threshold, gain) split for continuous targets."""
    # pylint: disable=too-many-locals
    n, n_features = int(X.shape[0]), int(X.shape[1])
    total_s, total_s2 = float(y.sum()), float((y * y).sum())
    parent_ss = total_s2 - total_s * total_s / n
    best = (0.0, -1, 0.0)
    for f in range(n_features):
        col = X[:, f]
        order = torch.argsort(col, stable=True)
        sorted_x, sorted_y = col[order], y[order]
        valid = sorted_x[:-1] != sorted_x[1:]
        if not bool(valid.any()):
            continue
        left_n = torch.arange(1, n, dtype=X.dtype, device=X.device)
        right_n = float(n) - left_n
        left_s = torch.cumsum(sorted_y, dim=0)[:-1]
        left_s2 = torch.cumsum(sorted_y * sorted_y, dim=0)[:-1]
        right_s, right_s2 = total_s - left_s, total_s2 - left_s2
        ss = left_s2 - left_s * left_s / left_n + right_s2 - right_s * right_s / right_n
        gain = torch.where(valid, parent_ss - ss, torch.zeros_like(ss))
        i = int(torch.argmax(gain).item())
        g = float(gain[i].item())
        if g > best[0]:
            best = (g, f, float((sorted_x[i] + sorted_x[i + 1]).item()) / 2.0)
    return best


class _BaseDecisionTree:
    """Shared CART builder storing the tree as parallel lists."""

    # pylint: disable=no-member

    def _validate_hyperparams(self, criteria) -> None:
        if self.criterion not in criteria:
            raise ValueError(
                f"criterion must be one of {criteria}, got {self.criterion!r}."
            )
        if self.max_depth is not None:
            if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
                raise TypeError(
                    "max_depth must be an int or None, "
                    f"got {type(self.max_depth).__name__}."
                )
            if int(self.max_depth) < 1:
                raise ValueError(f"max_depth must be >= 1, got {self.max_depth}.")
        if isinstance(self.min_samples_split, bool) or not isinstance(
            self.min_samples_split, int
        ):
            raise TypeError(
                "min_samples_split must be an int, "
                f"got {type(self.min_samples_split).__name__}."
            )
        if int(self.min_samples_split) < 2:
            raise ValueError(
                f"min_samples_split must be >= 2, got {self.min_samples_split}."
            )

    def _new_node(self, impurity, n_samples, value):
        """Append a node and return its id."""
        self._left.append(-1)
        self._right.append(-1)
        self._feature.append(-2)
        self._threshold.append(float("inf"))
        self._impurity.append(float(impurity))
        self._n_node_samples.append(int(n_samples))
        self._values.append(value)
        return len(self._left) - 1


class DecisionTreeClassifier(
    ClassifierMixin, _BaseDecisionTree
):  # pylint: disable=too-many-instance-attributes
    """CART classification tree (gini/entropy).

    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default='gini'
        Split impurity measure.
    max_depth : int or None, default=None
        Maximum tree depth. None means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split a node.

    Attributes
    ----------
    classes_ : torch.Tensor of shape (n_classes,)
        Sorted unique labels.
    feature_importances_ : torch.Tensor of shape (n_features,)
        Normalized impurity decrease per feature.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "DecisionTreeClassifier"

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        """Build the tree from ``X`` and ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted tree.
        """
        # pylint: disable=too-many-locals
        self._validate_hyperparams(("gini", "entropy"))
        from torml.utils._validation import check_X_y as _check_X_y

        X, y = _check_X_y(X, y)
        flat = y.reshape(-1)
        labels = flat.tolist()
        try:
            uniq = sorted(set(labels))
        except TypeError as e:
            raise TypeError("Labels must be sortable.") from e
        self.classes_ = (
            torch.as_tensor(uniq)
            if all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in uniq
            )
            else uniq
        )
        index_of = {c: i for i, c in enumerate(uniq)}
        y_idx = torch.tensor(
            [index_of[v] for v in labels], dtype=torch.long, device=X.device
        )
        n_classes = len(uniq)
        self.n_features_in_ = int(X.shape[1])
        self._left, self._right, self._feature = [], [], []
        self._threshold, self._impurity, self._n_node_samples, self._values = (
            [],
            [],
            [],
            [],
        )
        self._importance = torch.zeros(
            self.n_features_in_, dtype=X.dtype, device=X.device
        )

        max_depth = float("inf") if self.max_depth is None else int(self.max_depth)
        stack = [(0, torch.arange(int(X.shape[0]), device=X.device), 0)]
        self._new_node(0.0, int(X.shape[0]), None)
        while stack:
            node, idx, depth = stack.pop()
            node_y = y_idx[idx]
            counts = torch.bincount(node_y, minlength=n_classes).to(dtype=X.dtype)
            imp = _gini(counts) if self.criterion == "gini" else _entropy(counts)
            self._impurity[node] = float(imp.item())
            self._values[node] = counts
            n_node = int(idx.shape[0])
            if (
                depth >= max_depth
                or n_node < int(self.min_samples_split)
                or int((counts > 0).sum().item()) <= 1
            ):
                continue
            gain, feat, thr = _best_split_classifier(
                X[idx], node_y, n_classes, self.criterion
            )
            if feat < 0 or gain <= 0:
                continue
            left_mask = X[idx, feat] <= thr
            if not bool(left_mask.any()) or bool(left_mask.all()):
                continue
            self._feature[node] = feat
            self._threshold[node] = thr
            self._importance[feat] += gain
            left_id = self._new_node(0.0, int(left_mask.sum()), None)
            right_id = self._new_node(0.0, int((~left_mask).sum()), None)
            self._left[node], self._right[node] = left_id, right_id
            stack.append((right_id, idx[~left_mask], depth + 1))
            stack.append((left_id, idx[left_mask], depth + 1))

        total = float(self._importance.sum())
        self.feature_importances_ = (
            self._importance / total if total > 0 else self._importance
        )
        return self

    def _predict_idx(self, row) -> int:
        """Traverse to a leaf for one sample row."""
        node = 0
        while self._left[node] != -1:
            node = (
                self._left[node]
                if float(row[self._feature[node]]) <= self._threshold[node]
                else self._right[node]
            )
        return node

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
        check_is_fitted(self, attributes=["_left"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but the tree was fitted "
                f"with {int(self.n_features_in_)} features."
            )
        out = torch.empty(
            int(Xt.shape[0]),
            len(self._values[0]),
            dtype=Xt.dtype,
            device=Xt.device,
        )
        for i in range(int(Xt.shape[0])):
            counts = self._values[self._predict_idx(Xt[i])]
            out[i] = counts / counts.sum().clamp(min=1)
        return out

    def predict(self, X):
        """Predict class labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor or list of shape (n_samples,)
            Predicted labels.
        """
        proba = self.predict_proba(X)
        idx = torch.argmax(proba, dim=1)
        if isinstance(self.classes_, torch.Tensor):
            return self.classes_[idx]
        return [self.classes_[int(i)] for i in idx.tolist()]


class DecisionTreeRegressor(
    RegressorMixin, _BaseDecisionTree
):  # pylint: disable=too-many-instance-attributes
    """CART regression tree (squared error).

    Parameters
    ----------
    criterion : {'squared_error'}, default='squared_error'
        Split quality measure.
    max_depth : int or None, default=None
        Maximum tree depth. None means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split a node.

    Attributes
    ----------
    feature_importances_ : torch.Tensor of shape (n_features,)
        Normalized impurity decrease per feature.
    n_features_in_ : int
        Number of features seen during fit.
    """

    name = "DecisionTreeRegressor"

    def __init__(self, criterion="squared_error", max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        """Build the tree from ``X`` and ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted tree.
        """
        # pylint: disable=too-many-locals
        self._validate_hyperparams(("squared_error",))
        from torml.utils._validation import check_X_y as _check_X_y

        X, y = _check_X_y(X, y)
        target = y.to(dtype=X.dtype).reshape(-1)
        self.n_features_in_ = int(X.shape[1])
        self._left, self._right, self._feature = [], [], []
        self._threshold, self._impurity, self._n_node_samples, self._values = (
            [],
            [],
            [],
            [],
        )
        self._importance = torch.zeros(
            self.n_features_in_, dtype=X.dtype, device=X.device
        )

        max_depth = float("inf") if self.max_depth is None else int(self.max_depth)
        stack = [(0, torch.arange(int(X.shape[0]), device=X.device), 0)]
        self._new_node(0.0, int(X.shape[0]), 0.0)
        while stack:
            node, idx, depth = stack.pop()
            node_y = target[idx]
            mean = float(node_y.mean())
            ss = float(((node_y - mean) ** 2).sum())
            self._impurity[node] = ss
            self._values[node] = mean
            n_node = int(idx.shape[0])
            if depth >= max_depth or n_node < int(self.min_samples_split) or ss == 0:
                continue
            gain, feat, thr = _best_split_regressor(X[idx], node_y)
            if feat < 0 or gain <= 0:
                continue
            left_mask = X[idx, feat] <= thr
            if not bool(left_mask.any()) or bool(left_mask.all()):
                continue
            self._feature[node] = feat
            self._threshold[node] = thr
            self._importance[feat] += gain
            left_id = self._new_node(0.0, int(left_mask.sum()), 0.0)
            right_id = self._new_node(0.0, int((~left_mask).sum()), 0.0)
            self._left[node], self._right[node] = left_id, right_id
            stack.append((right_id, idx[~left_mask], depth + 1))
            stack.append((left_id, idx[left_mask], depth + 1))

        total = float(self._importance.sum())
        self.feature_importances_ = (
            self._importance / total if total > 0 else self._importance
        )
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
        check_is_fitted(self, attributes=["_left"])
        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but the tree was fitted "
                f"with {int(self.n_features_in_)} features."
            )
        out = torch.empty(int(Xt.shape[0]), dtype=Xt.dtype, device=Xt.device)
        for i in range(int(Xt.shape[0])):
            node = 0
            row = Xt[i]
            while self._left[node] != -1:
                node = (
                    self._left[node]
                    if float(row[self._feature[node]]) <= self._threshold[node]
                    else self._right[node]
                )
            out[i] = float(self._values[node])
        return out
