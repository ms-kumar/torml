# torml Release Notes

## [Unreleased]

### Added
- Added `torml.metrics.accuracy_score`, `torml.metrics.mean_squared_error`, `torml.metrics.r2_score` with tests in `tests/metrics/`.
- Added `torml.linear_model.LinearRegression` (closed-form via `torch.linalg.lstsq`) and `torml.linear_model.LogisticRegression` with tests in `tests/linear_model/`.
- Added `torml.model_selection.train_test_split`, `torml.model_selection.KFold`, and `torml.model_selection.cross_val_score` with tests in `tests/model_selection/`.
- Added `torml.preprocessing.StandardScaler`, `torml.preprocessing.MinMaxScaler`, `torml.preprocessing.LabelEncoder`, and `torml.preprocessing.OneHotEncoder` with tests in `tests/preprocessing/`.
- Added `torml.neighbors.KNeighborsClassifier` and `torml.neighbors.KNeighborsRegressor` (uniform/distance weights, Minkowski `p`, `kneighbors`, `predict_proba`) with tests in `tests/neighbors/`.
- Added `torml.naive_bayes.GaussianNB` (var smoothing, `predict_proba`/`predict_log_proba`, tensor and string labels) with tests in `tests/naive_bayes/`.
- Added `torml.cluster.KMeans` (Lloyd, best-of-`n_init` by inertia) and `torml.cluster.DBSCAN` (noise label -1) with tests in `tests/cluster/`.
- Added `torml.tree.DecisionTreeClassifier` (gini/entropy) and `torml.tree.DecisionTreeRegressor` (squared error) with tests in `tests/tree/`.
- Added `torml.decomposition.PCA` (full SVD, int/float-ratio `n_components`) with tests in `tests/decomposition/`.
- Added `torml.ensemble.VotingClassifier`/`VotingRegressor` (hard/soft vote, nested `name__param`), `torml.ensemble.BaggingClassifier`/`BaggingRegressor`, and `torml.ensemble.RandomForestClassifier`/`RandomForestRegressor` with tests in `tests/ensemble/`.
- Added `doc/` Sphinx site (`conf.py`, quickstart, user guide, API reference; builds with `sphinx-build`), runnable `examples/` per module, and `benchmarks/` skeleton.

### Changed
- Removed duplicate `torml/metrics.py`; `torml.metrics` package is the single source of truth.
- Removed `numpy` runtime dependency; torch-only backend (`torch.as_tensor`, `torch.linalg`).

### Fixed
- Fixed `BaseEstimator.get_params` to return `__init__` params with nested `__` support; fixed `clone(safe=False)` to deepcopy non-estimators.
- Fixed `check_array` dtype handling (torch.dtype and string), `check_X_y`, `check_is_fitted`, and `check_scalar` validation.
- Fixed `LinearRegression` to inherit `RegressorMixin` and implement correct `fit`/`predict` with `n_features_in_` validation.
- Fixed `BaseEstimator.score` to score without refitting (required for correct `cross_val_score`).

### Deprecated

### Removed

### Security

### Contributors
