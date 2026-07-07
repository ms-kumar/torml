"""Linear model module.

Provides linear regression and logistic regression estimators.
"""

from __future__ import annotations

import torch

import numpy as np

from torml.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from torml.utils import check_X_y, check_is_fitted


class LinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression.

    Linear regression fits a linear model using least squares.

    Read more in the :ref:`User Guide <linear_model>`.

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
        # Fit linear model w = (X*X')^-1 X*y

        # Reshape y to column vector
        y = y.reshape(-1, 1)

        X = X.t()

        # Compute intercept if fit_intercept is True
        if self.fit_intercept:
            X_mean = X.mean(dim=0, keepdim=True)
            y_mean = y.mean()
            X = X - X_mean.t()
            y = y - y_mean

        w, *_ = torch.linalg.lstsq(X, y)

        self.w_ = w.squeeze(1)
        if self.fit_intercept:
            self.intercept_ = y_mean.unsqueeze(0)
        else:
            self.intercept_ = None

        return self

    def _predict(self, X: torch.Tensor):
        """Apply linear model."""
        X = X.t()

        if self.fit_intercept:
            X_mean = X.mean(dim=0, keepdim=True)
            X = X - X_mean.t()

        y = X @ self.w_

        if self.fit_intercept:
            y = y + self.intercept_

        return y

    def predict(self, X: torch.Tensor):
        """Predict using the linear model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predictions.
        """
        return self._check_X(X)

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
            X_mean = X.mean(dim=0)
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


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression.

    Logistic regression fits a logistic model using maximum likelihood.

    Read more in the :ref:`User Guide <LogisticRegression>`.

    Parameters
    ----------
    penalty : str, default='l2'
        Penalty type. Must be one of 'l1', 'l2', 'elasticnet'.
    solver : str, default='lbfgs'
        Algorithm for optimization.

    Attributes
    ----------
    coef_ : ndarray of shape (n_in, n_out)
        Coefficients of the logistic model.
    intercept_ : ndarray of shape (n_out,)
        Intercept of logistic model.
    classes_ : ndarray of shape (n_classes,)
        Values of the target in the case of multi-class targets.

    Examples
    --------
    >>> from torml import LogisticRegression
    >>> import torch
    >>> X, y = torch.randn(10, 10), torch.randint(0, 2, (10, 5))
    >>> model = LogisticRegression(penalty='l2', solver='lbfgs')
    >>> model.fit(X, y)
    LogisticRegression(penalty='l2').solver='lbfgs'.fit(X, y)

    Notes
    -----
    This model represents a linear model for binary classification.

    The logistic regression minimizes a logistic loss over the training samples:

    .. math::

        (w,b) <- arg\\min_{w,b} \\sum_i L(y^{(i)}, <w, x^{(i)}> + b)

    where L is the logistic loss:

    .. math::

        L(v) = log(1 + \\exp(-y v)) + \\lambda_1 ||w||_2^2 + \\lambda_2 ||w||_1
    """

    def __init__(
        self,
        penalty: str = "l2",
        solver: str = "lbfgs",
    ):
        super().__init__()
        self.penalty = penalty
        self.solver = solver

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit Logistic Regression model.

        Uses Newton's method for optimization.

        Returns
        -------
        self : LogisticRegression instance
            Fitted classifier.
        """
    def _fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit Logistic Regression model using SGD.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training data.
        y : torch.Tensor of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LogisticRegression instance
            Fitted classifier.
        """
        # Initialize weights and bias
        self.coef_ = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
        self.intercept_ = torch.zeros(1, device=X.device, dtype=X.dtype)
        self.n_in_samples_ = X.shape[0]

        # SGD hyperparameters
        learning_rate = 0.01
        n_epochs = 100

        for epoch in range(n_epochs):
            # Shuffle data for SGD
            indices = torch.randperm(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices].unsqueeze(0)

            for i in range(X.shape[0]):
                x_i = X_shuffled[i]
                y_i = y_shuffled[i]

                # Predict probability of class 1
                linear_model = x_i @ self.coef_.t() + self.intercept_
                pred_proba = torch.sigmoid(linear_model)

                # Calculate loss gradient (for binary classification, this is equivalent to BCE derivative)
                error = pred_proba - y_i.unsqueeze(0) # Error term for SGD
                
                # Update weights and intercept
                self.coef_ -= learning_rate * error @ x_i.t()
                self.intercept_ -= learning_rate * torch.sigmoid(linear_model).mean() # Simplified update for intercept
                
            # Optional: Check convergence/loss here if needed

        return self

    def score(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute accuracy score on training data."""
        from torml.metrics import accuracy_score

        return accuracy_score(y, X @ self.coef_)