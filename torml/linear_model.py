"""Linear model module.

Provides linear regression and logistic regression estimators.
"""

from __future__ import annotations

import torch

from torml.base import ClassifierMixin, RegressorMixin
from torml.utils import check_is_fitted, check_X_y


class LinearRegression(RegressorMixin):
    """Linear regression.

    Linear regression fits a linear model using least squares.

    See the user guide for background on linear models.

    Parameters
    ----------
    fit_intercept : bool, default=True
        If True, fit linear model with intercept.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features, 1) if fit_intercept else (n_features,)
        Parameter vector ``w`` in equation :math:`y_i = w(X_i)`.
    intercept_ : ndarray of shape (1, 1) if fit_intercept else []
        Independent term in linear equation.
    n_in_samples_ : int
        Number of samples used to learn this regressor.
    n_features_in_ : int
        Number of features in the training data.

    Examples
    --------
    >>> from torml import LinearRegression
    >>> data = torch.randn(100, 10)
    >>> target = data @ torch.randn(10, 1)
    >>> model = LinearRegression(fit_intercept=False, random_state=None)
    >>> model.fit(data, target)
    LinearRegression(...).fit_intercept=False.random_state=None
    >>> model.coef_  # doctest: +SKIP
    array([...]

    Notes
    -----
    The linear regression model minimizes :math:'(y - X w)\\cdot(y - X w)'.
    The solution :math:`w` is obtained by using the closed form :math:`w = (X'X)^-1
    X'y`.

    The ``fit_intercept`` parameter does not affect :math:`w`, instead it
    fits the intercept term ``intercept_`` as the mean of y - Xw.

    The fit_intercept computation adds and subtracts the mean of the training
    data from each feature. This may result in values of X that are very large or
    very small. This may lead to large numerical errors in the coef_ estimates.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        normalize: bool = False,
    ):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit linear model with linear least squares.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples, 1)
            Target values.

        Returns
        -------
        self : LinearRegression instance
            Fitted regressor.
        """
        return self._fit(X, y)

    def _fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit linear model with linear least squares."""
        X, y = check_X_y(X, y)
        y = y.to(dtype=X.dtype).reshape(-1, 1)
        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
            X_aug = torch.cat([X, ones], dim=1)
        else:
            X_aug = X

        solution, *_ = torch.linalg.lstsq(X_aug, y)  # pylint: disable=not-callable
        solution = solution.squeeze(1)
        if self.fit_intercept:
            self.coef_ = solution[:-1]
            self.intercept_ = solution[-1:]
        else:
            self.coef_ = solution
            self.intercept_ = torch.zeros(1, dtype=X.dtype, device=X.device)

        return self

    def predict(self, X: torch.Tensor):
        """Predict using the linear model.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y : torch.Tensor of shape (n_samples,)
            Predictions.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        from torml.utils._validation import check_array

        X = check_array(X, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"was fitted with {self.n_features_in_} features."
            )
        return X @ self.coef_ + self.intercept_.squeeze()

    def _more_tags(self):
        return {"_estimator_type": "regressor"}

    def _validate_params(self):
        return True

    @staticmethod
    def _get_estimator_dtype(X: torch.Tensor) -> torch.dtype:
        """Get dtype of estimator from training data."""
        return X.dtype

    def _rescale_data(self, X: torch.Tensor):
        """Rescale data for numerical stability, returning X and the scaling."""
        X = X.clone()

        # Check features
        if isinstance(X.mean(dim=0), torch.Tensor):
            std = X.std(dim=0)
            scale = torch.reciprocal(std)
            X = X * scale

        return X

    def _invert_scaling(self, X: torch.Tensor):
        """Invert the data rescaling."""
        return None

    def _validate_data(
        self,
        X,
        y,
        validate_separately: bool = True,
        **check_params,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate data for fitting.

        Returns
        -------
        X, y : np.ndarray
            Processed input and target arrays.
        """
        return check_X_y(
            X,
            y,
            **check_params,
        )


class LogisticRegression(ClassifierMixin):
    """Logistic regression for binary classification.

    Fits a linear model by minimizing binary cross-entropy with L2
    regularization using full-batch gradient descent.

    Parameters
    ----------
    penalty : str, default='l2'
        Only 'l2' is supported.
    solver : str, default='lbfgs'
        Kept for API compatibility; optimization is gradient descent.
    C : float, default=1.0
        Inverse regularization strength. Must be > 0.
    max_iter : int, default=1000
        Gradient descent steps.

    Attributes
    ----------
    classes_ : torch.Tensor of shape (2,)
        Sorted class labels.
    coef_ : torch.Tensor of shape (n_features,)
        Weight vector.
    intercept_ : torch.Tensor of shape (1,)
        Bias term.
    n_features_in_ : int
        Number of features seen during fit.
    n_iter_ : int
        Steps actually run.

    Examples
    --------
    >>> import torch
    >>> from torml.linear_model import LogisticRegression
    >>> X = torch.randn(20, 2)
    >>> y = (X[:, 0] > 0).long()
    >>> model = LogisticRegression().fit(X, y)
    >>> model.predict(X[:3])
    """

    def __init__(
        self,
        penalty: str = "l2",
        solver: str = "lbfgs",
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        super().__init__()
        self.penalty = penalty
        self.solver = solver
        self.C = C
        self.max_iter = max_iter

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the logistic model with gradient descent.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training data.
        y : torch.Tensor of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : LogisticRegression instance
            Fitted classifier.
        """
        if self.penalty != "l2":
            raise ValueError(f"Only penalty='l2' is supported, got {self.penalty!r}.")
        if isinstance(self.C, bool) or not isinstance(self.C, (int, float)):
            raise TypeError(f"C must be a float, got {type(self.C).__name__}.")
        if float(self.C) <= 0:
            raise ValueError(f"C must be > 0, got {self.C}.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise TypeError(
                f"max_iter must be an int, got {type(self.max_iter).__name__}."
            )
        if int(self.max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        X, y = check_X_y(X, y)
        flat = y.reshape(-1)
        classes, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        if int(classes.shape[0]) != 2:
            raise ValueError(
                "LogisticRegression supports binary labels only, "
                f"got {int(classes.shape[0])} classes."
            )
        self.classes_ = classes
        target = inverse.to(dtype=X.dtype)
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        self.n_features_in_ = n_features
        lam = 1.0 / (n_samples * float(self.C))
        w = torch.zeros(n_features, dtype=X.dtype, device=X.device)
        b = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        for _ in range(int(self.max_iter)):
            logits = X @ w + b
            prob = torch.sigmoid(logits)
            error = (prob - target) / n_samples
            w = w - (X.T @ error + lam * w)
            b = b - error.sum()
        self.coef_ = w
        self.intercept_ = b.reshape(1)
        self.n_iter_ = int(self.max_iter)
        return self

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """Return signed logits for ``X``.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        scores : torch.Tensor of shape (n_samples,)
            Positive values predict ``classes_[1]``.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        from torml.utils._validation import check_array

        Xt = check_array(X, ensure_2d=True)
        if int(Xt.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"X has {int(Xt.shape[1])} features, but LogisticRegression "
                f"was fitted with {int(self.n_features_in_)} features."
            )
        return Xt @ self.coef_ + self.intercept_.squeeze()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for ``X``.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        proba : torch.Tensor of shape (n_samples, 2)
            Probabilities for ``classes_[0]`` and ``classes_[1]``.
        """
        proba_1 = torch.sigmoid(self.decision_function(X))
        return torch.stack([1 - proba_1, proba_1], dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for ``X``.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : torch.Tensor of shape (n_samples,)
            Predicted labels.
        """
        idx = (self.decision_function(X) >= 0).long()
        return self.classes_[idx]
