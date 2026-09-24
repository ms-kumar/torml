Supervised learning
===================

Estimators follow the scikit-learn shape: hyperparameters in ``__init__``,
``fit`` returns ``self``, learned attributes end with ``_``.

Regression with ``LinearRegression``
------------------------------------

Closed-form least squares via ``torch.linalg.lstsq``:

.. code-block:: python

   model = LinearRegression(fit_intercept=True).fit(X_train, y_train)
   y_pred = model.predict(X_test)

Classification with ``KNeighborsClassifier``
--------------------------------------------

Majority vote over the ``n_neighbors`` closest training points:

.. code-block:: python

   clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
   clf.predict(X_test)
   clf.predict_proba(X_test)

Scaling inputs
--------------

Fit the scaler on training data only, then apply to both splits:

.. code-block:: python

   scaler = StandardScaler().fit(X_train)
   X_train_s = scaler.transform(X_train)
   X_test_s = scaler.transform(X_test)

Validating with held-out data
-----------------------------

.. code-block:: python

   scores = cross_val_score(model, X, y, cv=5)
