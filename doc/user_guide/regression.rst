Regression
==========

Predict a continuous target per sample. Regressors implement ``predict``
and inherit ``score`` (R²) from ``RegressorMixin``.

Linear models
-------------

``LinearRegression`` solves least squares in closed form via
``torch.linalg.lstsq`` — exact, deterministic, no tuning:

.. code-block:: python

   from torml.linear_model import LinearRegression

   model = LinearRegression(fit_intercept=True).fit(X_train, y_train)
   y_pred = model.predict(X_test)   # X @ coef_ + intercept_

``LinearSVR`` instead fits an epsilon-insensitive tube (errors inside the
tube are free) with Pegasos updates; ``PLSRegression`` projects onto
latent directions first, which helps when features outnumber samples or
are highly correlated.

Non-linear models
-----------------

- ``KNeighborsRegressor``: averages the ``n_neighbors`` closest targets
  (``weights="distance"`` weights by proximity).
- ``DecisionTreeRegressor`` / ``RandomForestRegressor``: piecewise-constant
  fits; the forest averages trees to smooth the steps.
- ``GaussianProcessRegressor``: exact RBF inference that also returns
  uncertainty — ``predict(X, return_std=True)`` — and reports
  ``log_marginal_likelihood`` for kernel comparison.

Multiple targets
----------------

``MultiOutputRegressor`` fits one clone of any regressor per target
column:

.. code-block:: python

   from torml.multivariate import MultiOutputRegressor

   multi = MultiOutputRegressor(LinearRegression()).fit(X_train, Y_train)
   multi.predict(X_test)   # shape (n_samples, n_outputs)

Tips
----

- R² of 1 is perfect; 0 means "as good as predicting the mean"; negative
  means worse. Compare models with ``cross_val_score`` (see
  :doc:`model_selection`), not training score.
