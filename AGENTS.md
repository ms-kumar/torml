# AGENTS.md — torml Agent Guide

This repository is **torml**: a scikit-learn-style machine learning library implemented from scratch with **PyTorch** as the numerical backend.

## Source of Truth

- Follow `TORML_CODING_GUIDELINES.md` for repository structure, estimator API conventions, testing, documentation, and release-note policy.
- Do not treat existing practice/demo files as the target architecture unless explicitly asked.

## Core Mission

- Implement scikit-learn-style estimators and utilities in `torml/`.
- Use PyTorch (`torch.Tensor`, `torch.linalg`, `torch.testing`) instead of NumPy/SciPy for algorithm internals.
- Keep public APIs familiar to scikit-learn users: `fit`, `predict`, `transform`, `score`, `get_params`, `set_params`.

## Coding Rules

- Constructor arguments are hyperparameters only; store them unchanged on `self`.
- `fit` must return `self`.
- Learned attributes must end with `_` (`coef_`, `classes_`, `n_features_in_`).
- Validate inputs through shared utilities under `torml/utils/` instead of duplicating validation logic.
- Keep imports side-effect free; no printing, training, downloading, or heavy computation at import time.
- Prefer vectorized PyTorch tensor operations over Python loops.

## Testing Rules

- Add or update tests for every new estimator, metric, utility, or behavior change.
- Use `pytest` and `torch.testing.assert_close` for tensor comparisons.
- Test invalid input, not-fitted behavior, shape contracts, and basic numerical sanity.

## Documentation Rules

- Use NumPyDoc/scikit-learn-style docstrings for public classes and methods.
- Update docs under `doc/` for public API additions.
- Update `RELEASES.md` for user-visible changes.

## Agent Workflow

1. Read the relevant section of `TORML_CODING_GUIDELINES.md` before implementing.
2. Implement the smallest coherent change.
3. Add tests next to the relevant feature area.
4. Run the narrowest relevant tests first, then broader checks if available.
5. Do not introduce scikit-learn as an implementation dependency.
