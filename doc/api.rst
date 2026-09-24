API reference
=============

Base
----

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.base.BaseEstimator
   torml.base.clone

Utils
-----

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.utils.check_array
   torml.utils.check_X_y
   torml.utils.check_is_fitted
   torml.utils.check_random_state

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.metrics.accuracy_score
   torml.metrics.mean_squared_error
   torml.metrics.r2_score

Linear models
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.linear_model.LinearRegression
   torml.linear_model.LogisticRegression

Model selection
---------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.model_selection.train_test_split
   torml.model_selection.KFold
   torml.model_selection.cross_val_score

Preprocessing
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.preprocessing.StandardScaler
   torml.preprocessing.MinMaxScaler
   torml.preprocessing.LabelEncoder
   torml.preprocessing.OneHotEncoder

Neighbors
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.neighbors.KNeighborsClassifier
   torml.neighbors.KNeighborsRegressor

Naive Bayes
-----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.naive_bayes.GaussianNB

Clustering
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.cluster.KMeans
   torml.cluster.DBSCAN

Trees
-----

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.tree.DecisionTreeClassifier
   torml.tree.DecisionTreeRegressor

Decomposition
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.decomposition.PCA

Ensemble
--------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.ensemble.VotingClassifier
   torml.ensemble.VotingRegressor
   torml.ensemble.BaggingClassifier
   torml.ensemble.BaggingRegressor
   torml.ensemble.RandomForestClassifier
   torml.ensemble.RandomForestRegressor

Support vector machines
-----------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.svm.LinearSVC
   torml.svm.LinearSVR

Pipelines
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.pipelines.Pipeline
   torml.pipelines.FeatureUnion
   torml.pipelines.ColumnTransformer

Manifold
--------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.manifold.MDS

Gaussian processes
------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.gaussian_process.GaussianProcessRegressor
