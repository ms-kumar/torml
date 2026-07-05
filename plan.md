# torml: PyTorch-backed scikit-learn-style ML library — Implementation Plan

## Context

`torml` (`/Users/skumar/Space/work/torml`) is a greenfield project: only documentation/scaffolding existed at the start (`AGENTS.md`, `TORML_CODING_GUIDELINES.md`, `.github/` workflows, issue/PR templates, and four Copilot skill files). There was no git repository, no `pyproject.toml`, no `torml/` source package, and no `tests/` directory.

The intent (per `TORML_CODING_GUIDELINES.md` / `AGENTS.md`) is to reimplement a scikit-learn-style ML library from scratch, but using **PyTorch** (`torch.Tensor`, `torch.linalg`) as the numerical backend instead of NumPy/SciPy, while keeping scikit-learn's public API shape (`fit`/`predict`/`transform`/`score`/`get_params`/`set_params`), repo layout, testing conventions, and NumPyDoc-style documentation.

Full scikit-learn parity (linear_model, tree, ensemble, svm, cluster, naive_bayes, neighbors, decomposition, manifold, gaussian_process, mixture, multiclass, semi_supervised, covariance, cross_decomposition, feature_extraction, feature_selection, random_projection, discriminant analysis, pipelines, model_selection, preprocessing, metrics, datasets...) is a multi-week effort. This plan scopes the **first pass** to: repo bootstrap → core infra (`torml/base`, `torml/utils`) → one complete, tested, documented vertical slice (metrics → linear_model → model_selection basics). Everything beyond that is captured as an explicit backlog to execute the same way in follow-up passes.

Git will be initialized as a **local-only repository** for now (no remote). Commits happen **one per complete logical unit** (an estimator/utility/module together with its tests, docstrings, and `RELEASES.md` entry, only after local `pre-commit` + `pytest` pass) — mirroring the CI gate in `.github/workflows/ci.yml` and the checklist in `.github/pull_request_template.md`. No stub or partial-work commits.

---

## Phase 0 — Git & Repo Bootstrap

1. `git init` in `/Users/skumar/Space/work/torml` (default branch renamed to `main` to match the branch name referenced by `.github/workflows/ci.yml` and `test.yml`).
2. Add root tooling/config files (all gate `ci.yml`'s `pre-commit run --all-files` + `pytest`, or `test.yml`'s `pytest`, or `release.yml`'s `python -m build`):
   - `pyproject.toml` — `setuptools.build_meta` backend (needed because `MANIFEST.in` only setuptools honors); `requires-python = ">=3.10,<3.13"` (matches `test.yml` matrix); runtime dep `torch>=2.1` only (no numpy/scipy/scikit-learn, per `AGENTS.md`); `[project.optional-dependencies]`: `test = [pytest, pytest-cov]` (sufficient alone for `test.yml`), `dev = [pre-commit, black, isort, flake8, pylint, build]` (everything `.pre-commit-config.yaml` needs, since `ci.yml` installs `-e ".[dev,test]"` before running pre-commit), `doc = [sphinx, numpydoc, sphinx-rtd-theme]`; `[tool.black] line-length = 88`; `[tool.pytest.ini_options] testpaths = ["tests"]`.
   - `.pre-commit-config.yaml` — `pre-commit-hooks` (trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, check-merge-conflict), `black --line-length=88`, `isort --profile=black --line-length=88`, `flake8 --max-line-length=88 --extend-ignore=E203,W503`, `pylint` with `language: system` (it must import real installed deps including `torch`, which is too heavy to redeclare as a pre-commit `additional_dependencies` entry).
   - `.gitignore` (`__pycache__/`, `*.egg-info/`, `build/`, `dist/`, `.pytest_cache/`, `.coverage`, `htmlcov/`, `doc/_build/`, `.venv/`), `.gitattributes` (`* text=auto eol=lf`), `.editorconfig` (4-space Python indent, 88 max line length, LF, final newline).
   - `.flake8`, `.isort.cfg`, `.pylintrc`, `.coveragerc` — all pinned to the 88-char guideline max (79 preferred, 88 hard cap per `TORML_CODING_GUIDELINES.md`), consistent with black so lint/format never fight each other. `.pylintrc` needs `extension-pkg-allow-list=torch` (avoids false `no-member` on torch's C-extension attrs) and `good-names=X,y,X_train,X_test,y_train,y_test,n,i,j,k`.
   - `.python-version` — pins the interpreter (`3.11`) that `uv` provisions and uses for the project venv, resolving the mismatch between this machine's default `python3` (3.14.6) and the project's supported range.
   - `tox.ini` — local mirror of both CI workflows (`py310,py311,py312` test envs + a `lint` env running pre-commit), driven by `uv` via the `tox-uv` plugin (`requires = tox-uv` in `[tox]`) so tox uses `uv` for venv creation/installs instead of pip.
   - `.readthedocs.yaml` — builds `doc/conf.py` using the `doc` extra.
   - `MANIFEST.in` — includes `README.md`, `LICENSE`, `RELEASES.md`; excludes `tests/`, `benchmarks/`.
   - `LICENSE` (MIT), `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `CODEOWNERS` (placeholder team — needs a real GitHub handle before it's functional).
   - `CONTRIBUTING.md` — dev setup via `uv` (`uv sync --extra dev --extra test`, `uv run pre-commit install`), pointer to `AGENTS.md` + `.github/skills/*`, note that `pylint` needs the real env installed first.
   - `README.md` — pitch, install, quickstart `fit`/`predict` snippet, links to `doc/`, `CONTRIBUTING.md`, `RELEASES.md`.
   - `RELEASES.md` — seeded per the format already documented in `TORML_CODING_GUIDELINES.md`, with an initial `## [Unreleased]` section (empty `Added/Changed/Deprecated/Removed/Fixed/Security` subheadings) to accumulate entries as work lands.
3. Create empty `torml/` package root (`torml/__init__.py` reading `__version__` via `importlib.metadata.version("torml")`, no eager submodule imports) and `tests/__init__.py`, `tests/common/` (per the guideline's tree) so the package is importable and pytest has a home.
4. Install locally with `uv`: `uv sync --extra dev --extra test`, then `uv run pre-commit install`; verify `uv run pre-commit run --all-files` and `uv run pytest` (empty pass) both succeed — this is what CI will run. `uv sync` provisions the interpreter pinned in `.python-version` (`3.11`) itself — including downloading it if it isn't already installed — so it sidesteps this machine's unrelated default `python3` (3.14.6, outside the project's `>=3.10,<3.13` range and outside what PyTorch currently ships wheels for).
5. **Commit**: `chore: initialize repository scaffold and tooling` (first commit — everything above, including the pre-existing `AGENTS.md`/`TORML_CODING_GUIDELINES.md`/`.github/`, plus the `uv.lock` lockfile generated by `uv sync`).

## Phase 1 — Core Infra: `torml/base` and `torml/utils`

Every later estimator depends on this layer, so it's built and fully tested before any estimator module.

**`torml/base/__init__.py`**
- `BaseEstimator`: `get_params(deep=True)` (introspects `__init__` via `inspect.signature`), `set_params(**params)` (raises `ValueError` on unknown key, returns `self`), `__repr__(N_CHAR_MAX=700)`, `_get_tags()`.
- `clone(estimator, *, safe=True)` — module-level function; reconstructs an unfitted copy from `get_params(deep=False)`; raises `TypeError` for non-estimators when `safe=True`.
- Mixins: `ClassifierMixin` (`score` → `torml.metrics.accuracy_score`, imported lazily inside the method to avoid a base↔metrics circular import), `RegressorMixin` (`score` → `r2_score`, same lazy-import rule), `TransformerMixin` (`fit_transform` default = `fit(X,y).transform(X)`), `ClusterMixin` (`fit_predict` default = `fit(X); return self.labels_`).
- Free functions: `is_classifier`, `is_regressor`, `is_clusterer` (check `_estimator_type`).

**`torml/utils/_validation.py`**
- `check_array(array, *, dtype=torch.float32, ensure_2d=True, allow_nd=False, copy=False, force_all_finite=True, ensure_min_samples=1, ensure_min_features=1, device=None, input_name="X") -> torch.Tensor` — accepts tensor/ndarray/nested list via `torch.as_tensor`; `TypeError` for non-array-like input; **preserves caller's device** unless `device` is explicit; `ValueError` for wrong ndim, non-finite values, or below min samples/features.
- `check_X_y(X, y, **kwargs)` — `ValueError` if sample counts mismatch.
- `column_or_1d(y, *, warn=False)`.
- `check_is_fitted(estimator, attributes=None, *, msg=None)` — raises `NotFittedError` (subclass of `ValueError`, keeping to the `AGENTS.md` ValueError/TypeError split).
- `check_scalar(x, name, target_type, *, min_val=None, max_val=None, include_boundaries="both")` — `TypeError` for wrong type, `ValueError` for out-of-range.
- `has_fit_parameter(estimator, parameter)`.

**`torml/utils/_param_validation.py`**: `Interval`, `StrOptions`, `HasMethods`, `validate_parameter_constraints(...)` raising `InvalidParameterError(ValueError)`.

**`torml/utils/_random.py`**: `check_random_state(seed)` → `torch.Generator` (`None`/`int`/existing `Generator`; else `ValueError`).

**`torml/utils/_mask.py`**: `safe_mask(X, mask)`, `indices_to_mask(indices, n)`.

**`torml/utils/_tags.py`**: `_DEFAULT_TAGS`, `get_tags(estimator)`.

**`torml/utils/_estimator_checks.py`**: `check_estimator(estimator)` — generic conformance battery (get_params/set_params round-trip, repr doesn't raise, not-fitted calls raise `NotFittedError`, `fit` returns `self`, `n_features_in_` set, output shape correct); feeds `tests/common/_test_common.py`.

**`torml/utils/__init__.py`**: re-exports the public surface (`check_array`, `check_X_y`, `check_is_fitted`, `check_scalar`, `check_random_state`, `NotFittedError`, `safe_mask`, `get_tags`, `check_estimator`).

**`torml/_min_dependencies.py`**: `PYTHON_MIN_VERSION`, `TORCH_MIN_VERSION`, `PYTEST_MIN_VERSION`, `get_min_version(package)`.

**Tests** (`tests/base/test_base.py`, `tests/utils/test_validation.py`, `test_param_validation.py`, `test_random.py`, `test_mask.py`, `test_tags.py`, `test_estimator_checks.py`) covering: get/set_params round-trip + bad-key error, `clone` equality + `TypeError` on non-estimator, mixin `score`/`fit_transform`/`fit_predict` delegation, `is_classifier`/`is_regressor`/`is_clusterer`, `check_array` dtype/device/shape/finite behavior, `check_X_y` mismatch error, `check_is_fitted` pre/post fit, `check_scalar` type vs value errors, random-state reproducibility, mask correctness, tag merging, and `check_estimator` passing on a minimal dummy estimator of each kind. All using small deterministic CPU tensors + `torch.testing.assert_close`.

- **Commit**: `feat: add BaseEstimator API and shared validation utilities` (after full test pass + pre-commit clean).

## Phase 2 — First Vertical Slice

Establishes the working end-to-end pattern (implementation → tests → docs → `RELEASES.md`) that all later modules repeat. Each numbered item below is implemented and **committed separately**, per the "one commit per logical unit" convention:

1. **`torml/metrics/`** — `mean_squared_error`, `r2_score`, `accuracy_score` (needed by the base mixins' `score()` methods and by everything downstream). Tests in `tests/metrics/`.
2. **`torml/linear_model/`** — `LinearRegression` (closed-form via `torch.linalg.lstsq`), `LogisticRegression` (iterative solver, e.g. gradient descent or Newton via `torch.linalg`). Each: hyperparameters only in `__init__`, `fit` returns `self`, learned attrs `coef_`/`intercept_`/`n_features_in_` with trailing underscores, `predict` validates fitted state first. Tests in `tests/linear_model/` per the checklist in `.github/skills/validation-and-testing/SKILL.md`. NumPyDoc docstrings per `.github/skills/docs-and-release-notes/SKILL.md`.
3. **`torml/model_selection/`** — `train_test_split`, `KFold`, `cross_val_score` (enough to validate estimators against held-out data; `GridSearchCV` deferred to backlog). Tests in `tests/model_selection/`.

Each item: implement → add tests → add/update docstrings → add a `RELEASES.md` entry under `## [Unreleased]` naming the public symbol path (e.g. `torml.linear_model.LinearRegression`) → run `pre-commit run --all-files` + `pytest` locally → commit.

## Phase 3 — Backlog for Follow-Up Passes (tracked, not executed now)

Same procedure each time (the `.github/skills/implement-estimator`, `pytorch-backend`, `validation-and-testing`, `docs-and-release-notes` skill files already encode the checklist): implement → test → document → update `RELEASES.md` → commit per unit.

Suggested priority order for future passes:
1. `preprocessing` (StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder)
2. `neighbors` (KNeighborsClassifier/Regressor)
3. `naive_bayes` (GaussianNB)
4. `cluster` (KMeans, then DBSCAN)
5. `tree` (DecisionTreeClassifier/Regressor)
6. `decomposition` (PCA via `torch.linalg.svd`)
7. `ensemble` (RandomForest, Bagging, Voting — built on `tree`)
8. `svm` (start with a linear SVM before kernel/dual-solver variants)
9. `pipelines` (Pipeline, ColumnTransformer, FeatureUnion)
10. Remaining modules: `manifold`, `gaussian_process`, `mixture`, `multiclass`, `semi_supervised`, `covariance`, `cross_decomposition`, `feature_extraction`, `feature_selection`, `random_projection`, `discrimin_analysis`, `multivariate`
11. Full `doc/` Sphinx site (`conf.py`, `user_guide/*.rst`, `reference/*`), `examples/`, `benchmarks/` — once enough public API exists to document meaningfully.

## Git Workflow

- **Local repo only** — no remote configured in this pass. Add one later with `git remote add origin <url>` when ready to push/collaborate.
- **When to commit**: after each complete logical unit — implementation + tests + docstrings + `RELEASES.md` entry — and only once `pre-commit run --all-files` and `pytest` both pass locally (this is exactly what `.github/workflows/ci.yml` will run once a remote/CI exists). Never commit a stub, a half-implemented estimator, or a red test suite.
- **When to stage (`git add`)**: stage only the files belonging to that logical unit (name files explicitly; avoid `git add -A`/`git add .` so unrelated in-progress work isn't swept in).
- **Commit message convention**: short imperative prefix matching the change type (`feat:`, `fix:`, `test:`, `docs:`, `chore:`) — consistent with `RELEASES.md`'s Added/Changed/Fixed/Security categories — followed by the affected public path where applicable (e.g. `feat: implement torml.linear_model.LogisticRegression`).
- **PRs**: not applicable yet (no remote). Once a remote exists, follow `.github/pull_request_template.md`'s checklist (tests added, docs updated, release notes updated, no lint/type errors) before opening one.

## Verification

- After Phase 0: `uv sync --extra dev --extra test`, `uv run pre-commit run --all-files`, `uv run pytest` all succeed on the empty scaffold (mirrors what `ci.yml`/`test.yml` will run once those workflows are switched to `uv` too — they currently use plain `pip`, which still works against the same `pyproject.toml` since `uv` doesn't change the project's standard metadata).
- After Phase 1: `uv run pytest tests/base tests/utils -v` passes; `uv run python -c "import torml; from torml.base import BaseEstimator, clone; from torml.utils import check_array"` succeeds with no import-time side effects.
- After each Phase 2 item: run its narrow test file first (e.g. `uv run pytest tests/linear_model/ -v`), then the full suite (`uv run pytest`) and `uv run pre-commit run --all-files` before committing — matches the `AGENTS.md` Agent Workflow ("run the narrowest relevant tests first, then broader checks").
- Spot-check one estimator end-to-end interactively, e.g.:
  ```python
  import torch
  from torml.linear_model import LinearRegression
  X = torch.randn(50, 3)
  y = X @ torch.tensor([1.0, 2.0, -1.0]) + 0.1
  model = LinearRegression().fit(X, y)
  model.predict(X[:5])
  ```
