Supervised learning
===================

Supervised estimators learn from labeled pairs ``(X, y)``. They all follow
the same shape:

- Hyperparameters go in ``__init__`` and are stored unchanged.
- ``fit(X, y)`` learns and returns ``self`` (so calls chain).
- Learned attributes end with ``_``: ``coef_``, ``classes_``.
- ``predict(X)`` needs a fitted estimator, otherwise ``NotFittedError``.

.. toctree::
   :maxdepth: 1

   classification
   regression

Pick a page by task. Classification predicts discrete labels
(``KNeighborsClassifier``, ``LogisticRegression``, trees, forests, SVMs,
naive Bayes, LDA); regression predicts continuous targets
(``LinearRegression``, ``KNeighborsRegressor``, trees, GPs, PLS).
