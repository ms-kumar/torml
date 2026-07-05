---
name: implement-estimator
description: 'Use when: implementing torml estimators, classifiers, regressors, transformers, fit/predict/transform APIs, learned attributes, get_params, set_params, scikit-learn-style classes with PyTorch internals.'
argument-hint: '<estimator name or module>'
---

# Implement Estimator

Use this skill when adding or modifying a torml estimator.

## Procedure

1. Identify the estimator type: classifier, regressor, transformer, clusterer, metric utility, or model-selection helper.
2. Place implementation under the matching `torml/<module>/` package.
3. Keep `__init__` limited to hyperparameters and assign each to `self` unchanged.
4. Implement `fit(X, y=None)` and return `self`.
5. Store learned attributes with trailing underscores (`coef_`, `intercept_`, `classes_`, `n_features_in_`).
6. Implement prediction/transform methods only after fitted-state validation.
7. Use PyTorch tensor math (`torch`, `torch.linalg`) for algorithm internals.
8. Export public classes from the package `__init__.py`.
9. Add tests and docs before finishing.

## Required API Checks

- `fit` returns `self`.
- `get_params` and `set_params` work if inherited/implemented.
- Invalid shapes raise clear `ValueError`s.
- Calling predict/transform before fit raises a clear not-fitted error.

## PyTorch Rules

- Convert array-like inputs to `torch.Tensor` in validation utilities.
- Default dtype should be `torch.float32` unless the estimator needs otherwise.
- CPU should work by default.
- Device support must be explicit, not magical.
