Quickstart
==========

Install
-------

.. code-block:: bash

   pip install -e .

   # with test tools
   pip install -e ".[test]"

   # with docs tools
   pip install -e ".[doc]"

Regression
----------

.. code-block:: python

   import torch
   from torml.linear_model import LinearRegression

   X = torch.randn(50, 3)
   y = X @ torch.tensor([1.0, 2.0, -1.0]) + 0.1

   model = LinearRegression().fit(X, y)
   model.predict(X[:5])

Classification
--------------

.. code-block:: python

   import torch
   from torml.neighbors import KNeighborsClassifier

   X = torch.randn(40, 2)
   y = (X[:, 0] > 0).long()

   clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
   clf.predict(X[:5])

Preprocessing and validation
----------------------------

.. code-block:: python

   from torml.model_selection import cross_val_score, train_test_split
   from torml.preprocessing import StandardScaler

   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
   scaler = StandardScaler().fit(X_train)
   scores = cross_val_score(KNeighborsClassifier(), X_train, y_train, cv=3)
