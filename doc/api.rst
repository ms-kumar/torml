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

Mixture
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.mixture.GaussianMixture

Multiclass
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.multiclass.OneVsRestClassifier

Semi-supervised
---------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.semi_supervised.LabelPropagation

Covariance
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.covariance.EmpiricalCovariance

Cross decomposition
-------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.cross_decomposition.PLSRegression

Feature extraction
------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.feature_extraction.DictVectorizer

Feature selection
-----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.feature_selection.SelectKBest
   torml.feature_selection.f_classif

Random projection
-----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.random_projection.GaussianRandomProjection
   torml.random_projection.johnson_lindenstrauss_min_dim

Discriminant analysis
---------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.discriminant_analysis.LinearDiscriminantAnalysis

Multivariate outputs
--------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torml.multivariate.MultiOutputRegressor
   torml.multivariate.MultiOutputClassifier

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
