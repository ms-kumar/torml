API reference
=============

Every estimator follows the same conventions (see :doc:`user_guide/supervised`):
hyperparameters in ``__init__``, ``fit`` returns ``self``, learned
attributes end with ``_``.

Base
----

Estimator base class, cloning, and mixins.

.. autosummary::
   :toctree: _autosummary
   torml.base.BaseEstimator
   torml.base.clone

Utils
-----

Input validation, fitted-state checks, and random-state handling shared by
all estimators.

.. autosummary::
   :toctree: _autosummary
   torml.utils.check_array
   torml.utils.check_X_y
   torml.utils.check_is_fitted
   torml.utils.check_random_state

Metrics
-------

Scoring functions for classification and regression.

.. autosummary::
   :toctree: _autosummary
   torml.metrics.accuracy_score
   torml.metrics.mean_squared_error
   torml.metrics.r2_score

Linear models
-------------

Closed-form least squares and gradient-descent logistic regression.

.. autosummary::
   :toctree: _autosummary
   torml.linear_model.LinearRegression
   torml.linear_model.LogisticRegression

Model selection
---------------

Splitting, cross-validation, and exhaustive grid search.

.. autosummary::
   :toctree: _autosummary
   torml.model_selection.train_test_split
   torml.model_selection.KFold
   torml.model_selection.cross_val_score
   torml.model_selection.GridSearchCV

Preprocessing
-------------

Scaling, categorical encoding, and label encoding.

.. autosummary::
   :toctree: _autosummary
   torml.preprocessing.StandardScaler
   torml.preprocessing.MinMaxScaler
   torml.preprocessing.LabelEncoder
   torml.preprocessing.OneHotEncoder

Neighbors
---------

Vote/average over the nearest training points.

.. autosummary::
   :toctree: _autosummary
   torml.neighbors.KNeighborsClassifier
   torml.neighbors.KNeighborsRegressor

Naive Bayes
-----------

Gaussian likelihoods with per-class means and variances.

.. autosummary::
   :toctree: _autosummary
   torml.naive_bayes.GaussianNB

Mixture
-------

Full-covariance Gaussian mixtures fitted with EM.

.. autosummary::
   :toctree: _autosummary
   torml.mixture.GaussianMixture

Multiclass
----------

One-vs-rest wrapper turning binary estimators multiclass.

.. autosummary::
   :toctree: _autosummary
   torml.multiclass.OneVsRestClassifier

Semi-supervised
---------------

Label propagation over a kNN graph (transductive).

.. autosummary::
   :toctree: _autosummary
   torml.semi_supervised.LabelPropagation

Covariance
----------

Empirical mean/covariance/precision with Mahalanobis distances.

.. autosummary::
   :toctree: _autosummary
   torml.covariance.EmpiricalCovariance

Cross decomposition
-------------------

PLS regression onto latent directions that explain ``y``.

.. autosummary::
   :toctree: _autosummary
   torml.cross_decomposition.PLSRegression

Feature extraction
------------------

Dictionary rows to numeric matrices.

.. autosummary::
   :toctree: _autosummary
   torml.feature_extraction.DictVectorizer

Feature selection
-----------------

ANOVA F-scores with exact p-values, and top-k selection.

.. autosummary::
   :toctree: _autosummary
   torml.feature_selection.SelectKBest
   torml.feature_selection.f_classif

Random projection
-----------------

JL-lemma Gaussian projection for fast dimensionality cuts.

.. autosummary::
   :toctree: _autosummary
   torml.random_projection.GaussianRandomProjection
   torml.random_projection.johnson_lindenstrauss_min_dim

Discriminant analysis
---------------------

LDA classification with an optional discriminant projection.

.. autosummary::
   :toctree: _autosummary
   torml.discriminant_analysis.LinearDiscriminantAnalysis

Multivariate outputs
--------------------

One clone per target column, for regression and classification.

.. autosummary::
   :toctree: _autosummary
   torml.multivariate.MultiOutputRegressor
   torml.multivariate.MultiOutputClassifier

Clustering
----------

Center-based, density-based, and helper prediction methods.

.. autosummary::
   :toctree: _autosummary
   torml.cluster.KMeans
   torml.cluster.DBSCAN

Trees
-----

CART trees with gini/entropy/squared-error splits.

.. autosummary::
   :toctree: _autosummary
   torml.tree.DecisionTreeClassifier
   torml.tree.DecisionTreeRegressor

Decomposition
-------------

Full-SVD principal component analysis.

.. autosummary::
   :toctree: _autosummary
   torml.decomposition.PCA

Ensemble
--------

Voting, bagging, and random forests built on the estimators above.

.. autosummary::
   :toctree: _autosummary
   torml.ensemble.VotingClassifier
   torml.ensemble.VotingRegressor
   torml.ensemble.BaggingClassifier
   torml.ensemble.BaggingRegressor
   torml.ensemble.RandomForestClassifier
   torml.ensemble.RandomForestRegressor

Support vector machines
-----------------------

Pegasos sub-gradient linear SVMs for classification and regression.

.. autosummary::
   :toctree: _autosummary
   torml.svm.LinearSVC
   torml.svm.LinearSVR

Pipelines
---------

Chaining, concatenation, and column-wise composition.

.. autosummary::
   :toctree: _autosummary
   torml.pipelines.Pipeline
   torml.pipelines.FeatureUnion
   torml.pipelines.ColumnTransformer

Manifold
--------

Classical scaling preserving pairwise distances.

.. autosummary::
   :toctree: _autosummary
   torml.manifold.MDS

Gaussian processes
------------------

Exact RBF inference with posterior uncertainty.

.. autosummary::
   :toctree: _autosummary
   torml.gaussian_process.GaussianProcessRegressor

.. toctree::
   :hidden:
   :glob:

   _autosummary/*
