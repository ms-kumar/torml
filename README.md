# torml

[![CI](https://github.com/ms-kumar/torml/actions/workflows/ci.yml/badge.svg)](https://github.com/ms-kumar/torml/actions/workflows/ci.yml)
[![Tests](https://github.com/ms-kumar/torml/actions/workflows/test.yml/badge.svg)](https://github.com/ms-kumar/torml/actions/workflows/test.yml)
[![Bandit](https://github.com/ms-kumar/torml/actions/workflows/bandit.yml/badge.svg)](https://github.com/ms-kumar/torml/actions/workflows/bandit.yml)
[![PyPI](https://img.shields.io/pypi/v/torml.svg)](https://pypi.org/project/torml/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/ms-kumar/torml/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**torml** is a scikit-learn-style machine learning library implemented from
scratch with [PyTorch](https://pytorch.org/) (`torch.Tensor`, `torch.linalg`)
as its numerical backend — no NumPy, SciPy, or scikit-learn at runtime.

If you know scikit-learn, you know torml: estimators expose `fit` /
`predict` / `transform` / `score` / `get_params` / `set_params`, hyperparameters
live in `__init__`, `fit` returns `self`, and learned attributes end with `_`
(`coef_`, `classes_`, `n_features_in_`). The difference is what's underneath:
vectorized PyTorch tensor ops, so models compose with the torch ecosystem
without a second numerical stack.

## Highlights

- **25 estimator modules** — linear models, neighbors, trees, forests, SVMs,
  clustering, mixture models, naive Bayes, discriminant analysis, Gaussian
  processes, preprocessing, pipelines, model selection, metrics, and more.
- **280 tests**, typed and lint-clean — `black`, `isort`, `flake8`, `pylint`,
  Bandit, and CodeQL all enforced in CI across Python 3.10–3.12.
- **One runtime dependency**: `torch>=2.1`.
- **Security-first supply chain** — pinned lockfile (`uv.lock`), SHA-pinned
  GitHub Actions, least-privilege workflows, Dependabot + `pip-audit` clean.
  See [`SECURITY.md`](SECURITY.md).

## Installation

Requires Python 3.10–3.12:

```bash
pip install torml
```

With test and docs extras:

```bash
pip install "torml[test]"
pip install "torml[doc]"
```

From source (e.g. to follow `main`):

```bash
git clone https://github.com/ms-kumar/torml.git
cd torml
pip install -e .
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

## How we compare

Head-to-head against scikit-learn 1.9.1 (same data, same seeds, CPU;
rerun with `uv run --with scikit-learn python benchmarks/compare_sklearn.py`):

| task | metric | torml | scikit-learn |
| --- | --- | --- | --- |
| linreg | R² | 0.999656 | 0.999656 |
| kmeans | inertia (lower better) | 72645.4 | 72611.7 |
| knn | accuracy | 0.9476 | 0.9476 |
| tree | accuracy | 1.0000 | 1.0000 |
| scaler | max \|diff\| | 4.77e-07 | reference |

Correctness matches everywhere (10k samples per task); timings are in the
same class except tree fitting (~1.5x slower — Cython vs Python loops) and
10k kNN prediction, where sklearn's ball tree (~0.3s) beats torml's exact
brute force (~5s). torml's real edges are torch-native differentiable I/O
(including CUDA + float64 propagation), one runtime dependency, and exact
statistics. Full honest write-up (including where sklearn wins):
[user guide comparison](doc/user_guide/comparison.rst).

## Documentation

- API reference + user guide: [`doc/`](doc/) — build with
  `sphinx-build -b html doc doc/_build` (needs `uv sync --extra doc`)
- Wiki: https://github.com/ms-kumar/torml/wiki (conventions, validation utilities, module status)
- Examples: [`examples/`](examples/) — runnable scripts for every module
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
python -m build && twine check dist/*        # packaging check
```

## Security

See [`SECURITY.md`](SECURITY.md) for the supported-versions table and how to
report vulnerabilities privately. Dependencies are scanned by Dependabot and
`pip-audit`; CI pins all actions to commit SHAs.

## License

MIT — see [`LICENSE`](LICENSE).
