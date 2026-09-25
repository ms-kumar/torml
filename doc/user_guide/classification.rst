Classification
==============

Predict a discrete label per sample. All classifiers implement
``predict``; most also implement ``predict_proba`` (one column per class in
``classes_`` order) and inherit ``score`` (accuracy) from
``ClassifierMixin``.

Nearest neighbors
-----------------

``KNeighborsClassifier`` votes among the ``n_neighbors`` closest training
points (Minkowski distance of order ``p``). Good default baseline, no
training cost, but prediction scans the training set:

.. code-block:: python

   from torml.neighbors import KNeighborsClassifier

   clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
   clf.predict(X_test)          # labels
   clf.predict_proba(X_test)    # per-class probabilities

Use ``weights="distance"`` so closer neighbors count more.

Logistic regression
-------------------

``LogisticRegression`` fits a linear model by minimizing L2-regularized
binary cross-entropy with gradient descent. Fast and calibrated for
linearly separable data; binary labels only:

.. code-block:: python

   from torml.linear_model import LogisticRegression

   clf = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, y_train)
   clf.decision_function(X_test)  # signed distance from the boundary
   clf.predict_proba(X_test)      # sigmoid probabilities

Trees and forests
-----------------

``DecisionTreeClassifier`` learns axis-aligned splits (``criterion="gini"``
or ``"entropy"``); ``max_depth`` controls overfitting and
``feature_importances_`` shows what the tree used. Single trees overfit —
``RandomForestClassifier`` averages ``n_estimators`` trees trained on
bootstrap samples (set ``random_state`` for reproducibility):

.. code-block:: python

   from torml.ensemble import RandomForestClassifier

   clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)

Probabilistic and linear models
-------------------------------

- ``GaussianNB``: per-class Gaussian likelihoods; strong with small data.
- ``LinearDiscriminantAnalysis``: one linear boundary (binary) or
  ``n_classes - 1`` discriminant directions; also a projector via
  ``transform``.
- ``LinearSVC``: max-margin hyperplane (Pegasos updates); use
  ``decision_function`` to rank by confidence.

Multiclass and multi-output
---------------------------

Any binary estimator becomes multiclass with ``OneVsRestClassifier`` (one
clone per class, highest score wins). For several label columns at once,
``MultiOutputClassifier`` fits one clone per output:

.. code-block:: python

   from torml.multiclass import OneVsRestClassifier
   from torml.tree import DecisionTreeClassifier

   clf = OneVsRestClassifier(DecisionTreeClassifier()).fit(X_train, y_train)

Tips
----

- Scale features (see :doc:`preprocessing`) for distance- and
  gradient-based models (neighbors, SVMs, logistic regression); trees and
  naive Bayes don't need it.
- Start with ``cross_val_score`` (see :doc:`model_selection`) before tuning.
