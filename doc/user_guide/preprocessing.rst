Preprocessing
=============

Transformers implement ``fit``/``transform`` (and usually
``fit_transform``). The golden rule: **fit on training data only**, then
``transform`` both splits — fitting on test data leaks information.

Scaling
-------

- ``StandardScaler``: zero mean, unit variance per feature. Handles
  zero-variance columns (scale of 1, no division by zero) and inverts
  exactly with ``inverse_transform``.
- ``MinMaxScaler``: maps each feature into ``feature_range`` (default
  ``(0, 1)``); also invertible.

.. code-block:: python

   from torml.preprocessing import StandardScaler

   scaler = StandardScaler().fit(X_train)   # train only!
   X_train_s = scaler.transform(X_train)
   X_test_s = scaler.transform(X_test)

Encoding categories
-------------------

- ``LabelEncoder``: maps labels (tensors or strings) to ``0..n_classes-1``
  and back with ``inverse_transform``.
- ``OneHotEncoder``: expands each categorical column into binary columns
  (``handle_unknown="ignore"`` encodes unseen categories as zeros instead
  of raising).
- ``DictVectorizer``: turns rows of ``{name: value}`` dicts into a matrix
  with ``get_feature_names_out``.

Selecting and projecting features
---------------------------------

- ``SelectKBest``: keeps the ``k`` features with the highest ``f_classif``
  ANOVA scores (exact p-values included); ``get_support`` shows the mask.
- ``GaussianRandomProjection``: JL-lemma random projection for a fast,
  approximate dimensionality cut (``inverse_transform`` via pseudo-inverse).
- ``PLSRegression``: supervised alternative — latent directions chosen to
  explain ``y``, not just ``X`` variance.

Chaining
--------

``Pipeline`` strings steps together (``scaler -> model``) so
cross-validation (see :doc:`model_selection`) evaluates the whole chain
without leakage; ``ColumnTransformer`` applies different transformers to
different columns.
