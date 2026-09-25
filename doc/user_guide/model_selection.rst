Model selection
================

Split data, validate honestly, and tune hyperparameters.

Splitting
---------

``train_test_split`` shuffles (seeded by ``random_state``) and splits every
array identically. ``test_size``/``train_size`` accept fractions or counts;
``shuffle=False`` keeps order for sequential data:

.. code-block:: python

   from torml.model_selection import cross_val_score, train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

Cross-validation
----------------

``KFold`` yields ``(train_idx, test_idx)`` folds covering every sample
exactly once as test. ``cross_val_score`` clones the estimator per fold so
no state leaks; ``scoring`` accepts ``None`` (the estimator's own
``score``), ``"accuracy"``, ``"r2"``, ``"neg_mean_squared_error"``, or a
callable:

.. code-block:: python

   scores = cross_val_score(model, X, y, cv=5)   # one score per fold

Tuning
------

``GridSearchCV`` tries every combination in ``param_grid``, keeps
``best_params_``/``best_score_``/``cv_results_``, and refits the winner on
the full data (``refit=True``):

.. code-block:: python

   from torml.model_selection import GridSearchCV

   gs = GridSearchCV(model, {"n_neighbors": [1, 3, 7]}, cv=3).fit(X, y)
   gs.best_params_, gs.predict(X_test)

Tips
----

- Never tune on the test set: split off test data first, tune with CV on
  the train split, and evaluate once at the end.
- Put preprocessing inside a ``Pipeline`` (see :doc:`preprocessing`) so
  each fold fits its own scaler — otherwise statistics leak across folds.
