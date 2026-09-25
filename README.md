# torml

[![CI](https://github.com/ms-kumar/torml/actions/workflows/ci.yml/badge.svg)](https://github.com/ms-kumar/torml/actions/workflows/ci.yml)
[![Tests](https://github.com/ms-kumar/torml/actions/workflows/test.yml/badge.svg)](https://github.com/ms-kumar/torml/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/ms-kumar/torml/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green)](RELEASES.md)

**torml** is a scikit-learn-style machine learning library implemented from
scratch with [PyTorch](https://pytorch.org/) (`torch.Tensor`, `torch.linalg`)
as its numerical backend — no NumPy, SciPy, or scikit-learn at runtime.

If you know scikit-learn, you know torml: estimators expose `fit` /
`predict` / `transform` / `score` / `get_params` / `set_params`, hyperparameters
live in `__init__`, `fit` returns `self`, and learned attributes end with `_`
(`coef_`, `classes_`, `n_features_in_`). The difference is what's underneath:
vectorized PyTorch tensor ops, so models run on CPU (and compose with the
torch ecosystem) without a second numerical stack.

## Features

25 modules, 280 tests, typed and lint-clean (`black`, `isort`, `flake8`,
`pylint` all enforced in CI):

| Area | Modules |
| --- | --- |
| Core | `base` (`BaseEstimator`, `clone`, mixins), `utils` (validation, random state, tags) |
| Supervised | `linear_model`, `neighbors`, `naive_bayes`, `tree`, `svm`, `discriminant_analysis`, `gaussian_process` |
| Unsupervised | `cluster`, `mixture`, `decomposition`, `manifold` |
| Meta-estimators | `ensemble` (voting, bagging, forests), `multiclass`, `multivariate`, `pipelines` |
| Data & evaluation | `preprocessing`, `feature_extraction`, `feature_selection`, `random_projection`, `cross_decomposition`, `covariance`, `semi_supervised` |
| Evaluation | `metrics`, `model_selection` (`train_test_split`, `KFold`, `cross_val_score`, `GridSearchCV`) |

## Install

Requires Python 3.10–3.12 and PyTorch ≥ 2.1:

```bash
pip install -e .
pip install -e ".[test]"   # pytest
pip install -e ".[doc]"    # sphinx docs
```

With [uv](https://docs.astral.sh/uv/) (pinned interpreter + lockfile):

```bash
uv sync --extra dev --extra test
```

## Quickstart

Regression:

```python
import torch
from torml.linear_model import LinearRegression

X = torch.randn(50, 3)
y = X @ torch.tensor([1.0, 2.0, -1.0]) + 0.1

model = LinearRegression().fit(X, y)
model.predict(X[:5])
```

Classification with scaling and cross-validation:

```python
import torch
from torml.model_selection import cross_val_score, train_test_split
from torml.neighbors import KNeighborsClassifier
from torml.preprocessing import StandardScaler

X = torch.randn(100, 4)
y = (X[:, 0] > 0).long()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train = StandardScaler().fit_transform(X_train)

clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print(cross_val_score(clf, X_train, y_train, cv=3))
```

Pipelines compose the same pieces:

```python
from torml.pipelines import Pipeline

pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(5))]
).fit(X_train, y_train)
```

More in [`examples/`](examples/) (one runnable script per module) and the
[user guide](doc/user_guide/supervised.rst).

## Documentation

- API reference + user guide: [`doc/`](doc/) — build with
  `sphinx-build -b html doc doc/_build` (needs `uv sync --extra doc`)
- Wiki: https://github.com/ms-kumar/torml/wiki (conventions, validation utilities, module status)
- Examples: [`examples/`](examples/) — runnable scripts, verified in CI-adjacent local runs
- Benchmarks: [`benchmarks/`](benchmarks/) (micro-benchmarks, excluded from the sdist)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup, style, PR checklist
- Coding guidelines: [`TORML_CODING_GUIDELINES.md`](TORML_CODING_GUIDELINES.md)
- Release notes: [`RELEASES.md`](RELEASES.md) — see
  [v0.1.0](https://github.com/ms-kumar/torml/releases/tag/v0.1.0)

## Development

```bash
uv run pytest                        # 280 tests
uv run pre-commit run --all-files    # black, isort, flake8, pylint
uv run sphinx-build -b html doc doc/_build   # docs (zero warnings)
```

## Security

See [`SECURITY.md`](SECURITY.md) for the supported-versions table and how to
report vulnerabilities privately. Dependencies are scanned by Dependabot and
`pip-audit`; CI pins all actions to commit SHAs.

## License

MIT — see [`LICENSE`](LICENSE).
