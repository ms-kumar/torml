# torml Release Notes

## [Unreleased]

### Added

### Changed
- Changed `torml.model_selection.GridSearchCV` docs: added `examples/model_selection/plot_grid_search.py`.

### Deprecated

### Removed

### Fixed
- Fixed Bandit scan scope to `torml/` (test asserts no longer reported as B101).
- Fixed `check_estimator` name check (class name instead of full repr) and params round-trip comparison.
- Fixed `get_tags` mixin detection (was crashing on every estimator).
- Fixed `RandomState` attribute bookkeeping (reset/pickle/seed handling).
- Fixed `MultiOutputRegressor`/`MultiOutputClassifier` nested `estimator__param` support.

### Security
- Bumped `setuptools` 81.0.0 to 84.0.0 (CVE-2026-59890).
- Pinned GitHub Actions to commit SHAs; added Dependabot config and `SECURITY.md`.

### Contributors

## 0.1.0 - 2026-09-25

### Added
- Added `torml.metrics.accuracy_score`, `torml.metrics.mean_squared_error`, `torml.metrics.r2_score` with tests in `tests/metrics/`.
- Added `torml.linear_model.LinearRegression` (closed-form via `torch.linalg.lstsq`) and `torml.linear_model.LogisticRegression` with tests in `tests/linear_model/`.
- Added `torml.model_selection.train_test_split`, `torml.model_selection.KFold`, and `torml.model_selection.cross_val_score` with tests in `tests/model_selection/`.
- Added `torml.model_selection.GridSearchCV` (exhaustive grid, refit best) with tests in `tests/model_selection/`.
- Added `torml.preprocessing.StandardScaler`, `torml.preprocessing.MinMaxScaler`, `torml.preprocessing.LabelEncoder`, and `torml.preprocessing.OneHotEncoder` with tests in `tests/preprocessing/`.
- Added `torml.neighbors.KNeighborsClassifier` and `torml.neighbors.KNeighborsRegressor` (uniform/distance weights, Minkowski `p`, `kneighbors`, `predict_proba`) with tests in `tests/neighbors/`.
- Added `torml.naive_bayes.GaussianNB` (var smoothing, `predict_proba`/`predict_log_proba`, tensor and string labels) with tests in `tests/naive_bayes/`.
- Added `torml.mixture.GaussianMixture` (full-covariance EM) with tests in `tests/mixture/`.
- Added `torml.multiclass.OneVsRestClassifier` with tests in `tests/multiclass/`.
- Added `torml.semi_supervised.LabelPropagation` (kNN graph, hard clamping, transductive) with tests in `tests/semi_supervised/`.
- Added `torml.covariance.EmpiricalCovariance` (location/covariance/precision, Mahalanobis, log-likelihood score) with tests in `tests/covariance/`.
- Added `torml.cross_decomposition.PLSRegression` (NIPALS, single target) with tests in `tests/cross_decomposition/`.
- Added `torml.feature_extraction.DictVectorizer` with tests in `tests/feature_extraction/`.
- Added `torml.feature_selection.SelectKBest` with `f_classif` (ANOVA F + exact p-values) with tests in `tests/feature_selection/`.
- Added `torml.random_projection.GaussianRandomProjection` with `johnson_lindenstrauss_min_dim` with tests in `tests/random_projection/`.
- Added `torml.discriminant_analysis.LinearDiscriminantAnalysis` (SVD solver) with tests in `tests/discriminant_analysis/`.
- Added `torml.multivariate.MultiOutputRegressor` and `torml.multivariate.MultiOutputClassifier` with tests in `tests/multivariate/`.
- Added `torml.cluster.KMeans` (Lloyd, best-of-`n_init` by inertia) and `torml.cluster.DBSCAN` (noise label -1) with tests in `tests/cluster/`.
- Added `torml.tree.DecisionTreeClassifier` (gini/entropy) and `torml.tree.DecisionTreeRegressor` (squared error) with tests in `tests/tree/`.
- Added `torml.decomposition.PCA` (full SVD, int/float-ratio `n_components`) with tests in `tests/decomposition/`.
- Added `torml.ensemble.VotingClassifier`/`VotingRegressor` (hard/soft vote, nested `name__param`), `torml.ensemble.BaggingClassifier`/`BaggingRegressor`, and `torml.ensemble.RandomForestClassifier`/`RandomForestRegressor` with tests in `tests/ensemble/`.
- Added `torml.svm.LinearSVC` and `torml.svm.LinearSVR` (Pegasos sub-gradient) with tests in `tests/svm/`.
- Added `torml.pipelines.Pipeline` (nested `name__param`, predict/transform/score routing), `torml.pipelines.FeatureUnion`, and `torml.pipelines.ColumnTransformer` with tests in `tests/pipelines/`.
- Added `torml.manifold.MDS` (classical scaling, euclidean/precomputed) with tests in `tests/manifold/`.
- Added `torml.gaussian_process.GaussianProcessRegressor` (RBF, posterior std, LML) with tests in `tests/gaussian_process/`.
- Added `doc/` Sphinx site (`conf.py`, quickstart, user guide, API reference; builds with `sphinx-build`), runnable `examples/` per module, and `benchmarks/` skeleton.

### Changed
- Removed duplicate `torml/metrics.py`; `torml.metrics` package is the single source of truth.
- Removed `numpy` runtime dependency; torch-only backend (`torch.as_tensor`, `torch.linalg`).

### Fixed
- Fixed `BaseEstimator.get_params` to return `__init__` params with nested `__` support; fixed `clone(safe=False)` to deepcopy non-estimators.
- Fixed `check_array` dtype handling (torch.dtype and string), `check_X_y`, `check_is_fitted`, and `check_scalar` validation.
- Fixed `LinearRegression` to inherit `RegressorMixin` and implement correct `fit`/`predict` with `n_features_in_` validation.
- Fixed `LogisticRegression` (full-batch gradient descent on L2 binary cross-entropy, `C`/`max_iter`, `decision_function`/`predict_proba`, binary-only) with dedicated tests in `tests/linear_model/`.
- Fixed `BaseEstimator.score` to score without refitting (required for correct `cross_val_score`).

### Deprecated

### Removed

### Security

### Contributors
