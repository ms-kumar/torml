# torml Coding Guidelines & Repository Structure

## PLAN

1. Remove the `whats_new/` directory with all individual version files
2. Consolidate release notes into a single `RELEASES.md` file
3. Simplify the documentation structure
4. Keep the core repository structure (torml package, tests, docs)
5. Update the `whats_new` section to show it should link to Markdown release notes instead

---

## IMPLEMENTATION

## 📋 Overview

This document provides comprehensive guidelines for contributing to the torml project.
Follow these guidelines to ensure code quality, consistency, and maintainability.

**Project intent**

- Follow scikit-learn’s standards for API design, repository layout, documentation, and testing.
- Implement algorithms from scratch using PyTorch (`torch`) as the numerical backend (instead of NumPy/SciPy).

---

## 📁 Repository Structure

```
torml/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── question.md
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   └── test.yml
│   └── pull_request_template.md
├── benchmarks/
│   ├── _build.sh
│   ├── _plotting.sh
│   ├── _plotting.py
│   ├── regression/
│   │   ├── _plotting.py
│   │   └── test_plotting.py
│   ├── clustering/
│   │   └── _plotting.py
│   ├── classification/
│   │   └── _plotting.py
│   └── model_selection/
│       └── _plotting.py
├── doc/
│   ├── _templates/
│   ├── _static/
│   ├── _autosummary/
│   ├── conf.py
│   ├── genindex.rst
│   ├── searchindex.js
│   ├── user_guide/
│   │   ├── 01_introduction.rst
│   │   ├── 02_overview.rst
│   │   ├── 03_terminology.rst
│   │   ├── 04_01_data.rst
│   │   ├── 04_02_data_matrix.rst
│   │   ├── 05_01_supervised.rst
│   │   ├── 05_02_supervised_classification.rst
│   │   ├── 05_03_supervised_regression.rst
│   │   ├── 06_01_unsupervised.rst
│   │   ├── 06_02_unsupervised_clustering.rst
│   │   ├── 07_01_preprocessing.rst
│   │   ├── 07_02_preprocessing_pipeline.rst
│   │   ├── 08_01_model_selection.rst
│   │   ├── 08_02_model_selection_cross_val.rst
│   │   ├── 08_03_model_selection_cross_val_cv.rst
│   │   ├── 08_04_model_selection_grid_search.rst
│   │   ├── 08_05_model_selection_random_search.rst
│   │   ├── 08_06_model_selection_parameter_grid.rst
│   │   ├── 08_07_model_selection_parameter_sampling.rst
│   │   ├── 08_08_model_selection_predefined_split.rst
│   │   ├── 09_01_meta_estimators.rst
│   │   ├── 09_02_meta_estimators_voting.rst
│   │   ├── 09_03_meta_estimators_pipeline.rst
│   │   ├── 09_04_meta_estimators_column.rst
│   │   ├── 09_05_meta_estimators_feature_union.rst
│   │   ├── 09_06_meta_estimators_column_transformer.rst
│   │   ├── 10_01_ensemble.rst
│   │   ├── 10_02_ensemble_bagging.rst
│   │   ├── 10_03_ensemble_boosting.rst
│   │   ├── 10_04_ensemble_extra_trees.rst
│   │   ├── 10_05_ensemble_stacking.rst
│   │   ├── 10_06_ensemble_voting.rst
│   │   ├── 10_07_ensemble_weighted_knn.rst
│   │   ├── 10_08_ensemble_random_forest.rst
│   │   ├── 11_01_nearest_neighbour.rst
│   │   ├── 11_02_nearest_centroid.rst
│   │   ├── 12_01_naive_bayes.rst
│   │   ├── 12_02_naive_bayes_multinomial.rst
│   │   ├── 12_03_naive_bayes_categorical.rst
│   │   ├── 12_04_naive_bayes_gaussian_mixture.rst
│   │   ├── 12_05_naive_bayes_gaussian_process.rst
│   │   ├── 12_06_naive_bayes_multinomial_gaussian.rst
│   │   ├── 13_01_svm.rst
│   │   ├── 13_02_svm_rbf.rst
│   │   ├── 13_03_svm_poly.rst
│   │   ├── 13_04_svm_sigmoid.rst
│   │   ├── 14_01_mlp.rst
│   │   ├── 14_02_mlp_multioutput.rst
│   │   ├── 14_03_mlp_elman.rst
│   │   ├── 15_01_gaussian_process.rst
│   │   ├── 15_02_gaussian_process_regression.rst
│   │   ├── 15_03_gaussian_process_classification.rst
│   │   ├── 16_01_decision_tree.rst
│   │   ├── 16_02_decision_tree_regression.rst
│   │   ├── 17_01_discriminative_analysis.rst
│   │   ├── 17_02_discriminative_analysis_logistic.rst
│   │   ├── 17_03_discriminative_analysis_mixture.rst
│   │   ├── 17_04_discriminative_analysis_naive_bayes.rst
│   │   ├── 17_05_discriminative_analysis_perceptron.rst
│   │   ├── 17_06_discriminative_analysis_svm.rst
│   │   ├── 18_01_discriminative_analysis_discriminant_analysis.rst
│   │   ├── 18_02_discriminative_analysis_discriminant_analysis_binary.rst
│   │   ├── 18_03_discriminative_analysis_discriminant_analysis_multiclass.rst
│   │   ├── 18_04_discriminative_analysis_discriminant_analysis_multinomial.rst
│   │   ├── 19_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   │   ├── 20_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   │   ├── 21_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   │   ├── 22_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   │   └── 23_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   ├── reference/
│   │   ├── all/
│   │   │   └── _autosummary/
│   │   ├── clustering/
│   │   ├── classification/
│   │   ├── datasets/
│   │   ├── inspection/
│   │   ├── linear_model/
│   │   ├── manifold/
│   │   ├── metrics/
│   │   ├── model_selection/
│   │   ├── naive_bayes/
│   │   ├── neighbors/
│   │   ├── preprocessing/
│   │   ├── svm/
│   │   └── tree/
│   └── whats_new/
│       └── README.md
├── examples/
│   ├── app/
│   ├── classification/
│   ├── clustering/
│   ├── datasets/
│   ├── linear_model/
│   ├── manifold/
│   ├── model_selection/
│   ├── naive_bayes/
│   ├── neighbors/
│   ├── preprocessing/
│   ├── svm/
│   └── tree/
├── torml/
│   ├── _distractors/
│   ├── _distutils/
│   ├── _isotonic/
│   ├── _min_dependencies.py
│   ├── _monitor/
│   ├── _plotting/
│   ├── _sample/
│   ├── _utils/
│   ├── base/
│   │   └── __init__.py
│   ├── cluster/
│   │   ├── _bicluster.py
│   │   ├── _dbscan.py
│   │   ├── _hierarchical.py
│   │   ├── _kmeans.py
│   │   ├── _kmeans_fast.py
│   │   ├── _kmeans_common.py
│   │   ├── _kmeansh.py
│   │   ├── _optics.py
│   │   ├── _spectral.py
│   │   ├── _birch.py
│   │   ├── _feature_agglomeration.py
│   │   ├── _kmeans_dbscan.py
│   │   └── __init__.py
│   ├── covariance/
│   ├── cross_decomposition/
│   ├── datasets/
│   ├── decomposition/
│   ├── discrimin_analysis/
│   ├── ensemble/
│   ├── feature_extraction/
│   ├── feature_selection/
│   ├── gaussian_process/
│   ├── linear_model/
│   ├── manifold/
│   ├── metrics/
│   ├── mixture/
│   ├── model_selection/
│   ├── multiclass/
│   ├── multivariate/
│   ├── naive_bayes/
│   ├── neighbors/
│   ├── pipelines/
│   ├── preprocessing/
│   ├── random_projection/
│   ├── semi_supervised/
│   ├── svm/
│   ├── tree/
│   ├── utils/
│   │   ├── _estimator_checks.py
│   │   ├── _mask.py
│   │   ├── _param_validation.py
│   │   ├── _random.py
│   │   ├── _tags.py
│   │   ├── _validation.py
│   │   └── __init__.py
│   └── __init__.py
├── RELEASES.md
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CODEOWNERS
├── MANIFEST.in
├── tox.ini
├── .pre-commit-config.yaml
├── .readthedocs.yaml
├── .gitignore
├── .gitattributes
├── .editorconfig
├── .pylintrc
├── .flake8
├── .isort.cfg
├── .coveragerc
└── tests/
    ├── __init__.py
    ├── common/
    │   ├── _config.py
    │   ├── _tags.py
    │   ├── _test_common.py
    │   └── _test_common.py
    ├── benchmarks/
    ├── cluster/
    ├── covariates/
    ├── cross_decomposition/
    ├── datasets/
    ├── decomposition/
    ├── discrimin_analysis/
    ├── ensemble/
    ├── feature_extraction/
    ├── feature_selection/
    ├── gaussian_process/
    ├── linear_model/
    ├── manifold/
    ├── metrics/
    ├── mixture/
    ├── model_selection/
    ├── multiclass/
    ├── multivariate/
    ├── naive_bayes/
    ├── neighbors/
    ├── pipelines/
    ├── preprocessing/
    ├── random_projection/
    ├── semi_supervised/
    ├── svm/
    └── tree/
```

---

## 📄 RELEASE NOTES

### Release Notes Format

All release notes should be consolidated in `RELEASES.md` instead of individual version files.

```markdown
# torml Release Notes

## [Version X.Y.Z] - [Release Date]

### Added
- [Feature 1 description]
- [Feature 2 description]
- [Feature 3 description]

### Changed
- [Change 1 description]
- [Change 2 description]
- [Change 3 description]

### Deprecated
- [Deprecation 1 description]
- [Deprecation 2 description]

### Removed
- [Removal 1 description]
- [Removal 2 description]

### Fixed
- [Bug fix 1 description]
- [Bug fix 2 description]
- [Bug fix 3 description]

### Security
- [Security fix 1 description]
- [Security fix 2 description]

### Contributors
- [Name]
- [Name]
- [Name]
```

### Example Release Notes Entry

```markdown
## 0.1.0 - 2026-07-05

### Added
- Added `torml.linear_model.LogisticRegression` with a PyTorch-based solver
- Added `torml.utils._validation.check_X_y`
- Added basic documentation skeleton under `doc/`

### Changed
- Updated estimator base class to store `n_features_in_`

### Fixed
- Fixed shape checking for 1D tensors

### Security
- [Security fix 1 description]

### Contributors
- Jane Doe
- John Smith
- Alice Johnson
```

---

## 🎯 Code Style Guidelines

### 1. Python Version & Compatibility

```python
# ✅ DO: Use modern Python features
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional, Protocol

import torch
```

### 2. Import Organization

```python
# ✅ DO: Organize imports in this order
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import os
import sys

# 3. Third-party packages
import torch

# 4. Local imports (alphabetical)
from torml.base import BaseEstimator, ClassifierMixin
from torml.utils._validation import check_array
from torml.metrics import mean_squared_error
```

### 3. Naming Conventions

```python
# ✅ DO: Follow these conventions

# Public classes/functions
def calculate_metric():
    pass


class MyEstimator(BaseEstimator):
    pass


# Private methods
def _private_helper():
    pass


# Constants
MAX_ITERATIONS = 100
EPSILON = 1e-8
```

### 4. Type Hints (Recommended)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import torch

if TYPE_CHECKING:
    from torml.base import BaseEstimator


class MyEstimator(BaseEstimator):
    """My custom estimator."""

    def __init__(
        self,
        *,
        param1: float = 0.5,
        param2: Literal["option1", "option2"] = "option1",
        param3: Optional[Union[int, float]] = None,
    ):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "MyEstimator":
        """Fit the model.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input features.
        y : torch.Tensor of shape (n_samples,)
            Target values.

        Returns
        -------
        self : MyEstimator
            Fitted estimator.
        """
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        raise NotImplementedError
```

### 5. Docstring Format (NumPyDoc / scikit-learn style)

```python
"""
{summary}

{extended_summary}

Parameters
----------
{param_name} : {type}
    {param_description}

    .. note::
        {note_description}

    .. warning::
        {warning_description}

    .. versionadded:: 0.1
        {version_description}

Returns
-------
{return_type} : {type}
    {return_description}

Raises
------
{exception_name} : {exception_type}
    {exception_description}

Examples
--------
>>> {example_code}
{example_output}

See Also
--------
{related_symbol} : {related_module}.{related_symbol}
    {related_symbol_description}

Notes
-----
{notes}

References
----------
{references}

References
----------
1. {author}. {title}. {journal}, {year}.
2. {author}. {title}. {URL}, {year}.
"""
```

### 6. Code Formatting

```python
# ✅ DO: Follow PEP8 with project-specific rules

# Line length: 79 characters (max 88 for long lines)
def very_long_function_name():
    pass


# Indentation: 4 spaces (no tabs)
def my_function():
    if True:
        do_something()


# Blank lines: 2 blank lines between top-level definitions
class MyClass:
    pass


def another_function():
    pass
```

### 7. Code Quality

```python
# ✅ DO: Use meaningful variable names and vectorized tensor ops

def calculate_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    diff = y_true - y_pred
    return (diff * diff).mean()
```

### 8. Error Handling

```python
# ✅ DO: Raise meaningful exceptions

def fit(self, X: torch.Tensor, y: torch.Tensor):
    if X is None:
        raise ValueError("X cannot be None")
    if y is None:
        raise ValueError("y cannot be None")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
```

---

## 🧪 Testing Guidelines

### 1. Test Structure

```python
# test_module_name.py
import pytest
import torch


class TestClassName:
    """Test class for module."""

    def test_method_name(self):
        pass

    def test_method_name_edge_case(self):
        pass

    def test_method_name_error_handling(self):
        pass
```

### 2. Test Templates

```python
"""Module tests."""

import pytest
import torch


class TestEstimator:
    @pytest.fixture
    def data(self):
        X = torch.randn(10, 5)
        y = torch.randn(10)
        return {"X": X, "y": y}

    def test_fit(self, data):
        pass

    def test_predict(self, data):
        pass
```

### 3. Test Assertions

```python
# ✅ DO: Use these assertions

# For numerical equality / tolerance
torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

# For exact tensor equality
assert torch.equal(actual, expected)

# For shape equality
assert tuple(actual.shape) == tuple(expected.shape)

# For None checks
assert result is None

# For type checks
assert isinstance(result, MyClass)

# For truthiness
assert result is True

# For exception testing
with pytest.raises(ValueError, match="expected message"):
    function_that_raises_error()

# For custom project assertions (recommended pattern)
# - Prefer a small `torml.utils._testing` module that wraps torch + pytest patterns
# - Keep it dependency-light and stable
```

### 4. Test Parameters

```python
import pytest


@pytest.mark.parametrize(
    "param, expected",
    [
        (1, 2),
        (2, 4),
        (3, 6),
    ],
)
def test_function(param, expected):
    assert param * 2 == expected
```

---

## 📝 Documentation Guidelines

### 1. Documentation Structure

```
doc/
├── _templates/
├── _static/
├── _autosummary/
├── conf.py
├── genindex.rst
├── searchindex.js
├── user_guide/
│   ├── 01_introduction.rst
│   ├── 02_overview.rst
│   ├── 03_terminology.rst
│   ├── 04_01_data.rst
│   ├── 04_02_data_matrix.rst
│   ├── 05_01_supervised.rst
│   ├── 05_02_supervised_classification.rst
│   ├── 05_03_supervised_regression.rst
│   ├── 06_01_unsupervised.rst
│   ├── 06_02_unsupervised_clustering.rst
│   ├── 07_01_preprocessing.rst
│   ├── 07_02_preprocessing_pipeline.rst
│   ├── 08_01_model_selection.rst
│   ├── 08_02_model_selection_cross_val.rst
│   ├── 08_03_model_selection_cross_val_cv.rst
│   ├── 08_04_model_selection_grid_search.rst
│   ├── 08_05_model_selection_random_search.rst
│   ├── 08_06_model_selection_parameter_grid.rst
│   ├── 08_07_model_selection_parameter_sampling.rst
│   ├── 08_08_model_selection_predefined_split.rst
│   ├── 09_01_meta_estimators.rst
│   ├── 09_02_meta_estimators_voting.rst
│   ├── 09_03_meta_estimators_pipeline.rst
│   ├── 09_04_meta_estimators_column.rst
│   ├── 09_05_meta_estimators_feature_union.rst
│   ├── 09_06_meta_estimators_column_transformer.rst
│   ├── 10_01_ensemble.rst
│   ├── 10_02_ensemble_bagging.rst
│   ├── 10_03_ensemble_boosting.rst
│   ├── 10_04_ensemble_extra_trees.rst
│   ├── 10_05_ensemble_stacking.rst
│   ├── 10_06_ensemble_voting.rst
│   ├── 10_07_ensemble_weighted_knn.rst
│   ├── 10_08_ensemble_random_forest.rst
│   ├── 11_01_nearest_neighbour.rst
│   ├── 11_02_nearest_centroid.rst
│   ├── 12_01_naive_bayes.rst
│   ├── 12_02_naive_bayes_multinomial.rst
│   ├── 12_03_naive_bayes_categorical.rst
│   ├── 12_04_naive_bayes_gaussian_mixture.rst
│   ├── 12_05_naive_bayes_gaussian_process.rst
│   ├── 12_06_naive_bayes_multinomial_gaussian.rst
│   ├── 13_01_svm.rst
│   ├── 13_02_svm_rbf.rst
│   ├── 13_03_svm_poly.rst
│   ├── 13_04_svm_sigmoid.rst
│   ├── 14_01_mlp.rst
│   ├── 14_02_mlp_multioutput.rst
│   ├── 14_03_mlp_elman.rst
│   ├── 15_01_gaussian_process.rst
│   ├── 15_02_gaussian_process_regression.rst
│   ├── 15_03_gaussian_process_classification.rst
│   ├── 16_01_decision_tree.rst
│   ├── 16_02_decision_tree_regression.rst
│   ├── 17_01_discriminative_analysis.rst
│   ├── 17_02_discriminative_analysis_logistic.rst
│   ├── 17_03_discriminative_analysis_mixture.rst
│   ├── 17_04_discriminative_analysis_naive_bayes.rst
│   ├── 17_05_discriminative_analysis_perceptron.rst
│   ├── 17_06_discriminative_analysis_svm.rst
│   ├── 18_01_discriminative_analysis_discriminant_analysis.rst
│   ├── 18_02_discriminative_analysis_discriminant_analysis_binary.rst
│   ├── 18_03_discriminative_analysis_discriminant_analysis_multiclass.rst
│   ├── 18_04_discriminative_analysis_discriminant_analysis_multinomial.rst
│   ├── 19_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   ├── 20_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   ├── 21_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   ├── 22_01_discriminative_analysis_discriminant_analysis_multinomial.rst
│   └── 23_01_discriminative_analysis_discriminant_analysis_multinomial.rst
├── reference/
│   ├── all/
│   │   └── _autosummary/
│   ├── clustering/
│   ├── classification/
│   ├── datasets/
│   ├── inspection/
│   ├── linear_model/
│   ├── manifold/
│   ├── metrics/
│   ├── model_selection/
│   ├── naive_bayes/
│   ├── neighbors/
│   ├── preprocessing/
│   ├── svm/
│   └── tree/
└── RELEASES.md
```

---

## 📋 Contribution Guidelines

### 1. Before Submitting

- [ ] Run tests locally
- [ ] Check code style with `pre-commit` hooks
- [ ] Update documentation if needed
- [ ] Add/update tests for new features
- [ ] Update release notes in `RELEASES.md`

### 2. Pull Request Template

```markdown
## Description

[Describe what this PR does]

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other: [Specify]

## Testing

[Describe how you tested this change]

## Screenshots

[If applicable, add screenshots]

## Checklist

- [ ] Code follows torml style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Release notes updated
- [ ] No linting errors
- [ ] No type errors

## Related Issues

Closes #[issue_number]
```

---

## 🛡️ Quality Checklist

- [ ] Is the plan logical and appropriate?
- [ ] Are edge cases considered?
- [ ] Is the output complete (no stubs unless requested)?
- [ ] Are coding standards met?
- [ ] Is the explanation clear and helpful?
- [ ] Have you verified the content accuracy?
- [ ] Tests pass locally
- [ ] Code style checks pass
- [ ] Documentation is accurate
- [ ] Release notes are updated

---

## 📊 Performance Guidelines

### 1. Memory Usage

```python
# ✅ DO: Use memory-efficient operations
X = X.to(dtype=torch.float32)

# ❌ DON'T: Create unnecessary copies
X = torch.tensor(X)
X = X.to(dtype=torch.float32)
```

### 2. Time Complexity

```python
# ✅ DO: Aim for O(n) or better
def efficient_algorithm(data: torch.Tensor) -> torch.Tensor:
    return data + 1

# ❌ DON'T: Use O(n²) when O(n) is possible
def inefficient_algorithm(data: torch.Tensor) -> list[tuple[int, int]]:
    pairs = []
    n = int(data.shape[0])
    for i in range(n):
        for j in range(n):
            if float(data[i]) == float(data[j]):
                pairs.append((i, j))
    return pairs
```

### 3. Vectorization

```python
# ✅ DO: Use vectorized operations
result = (X ** 2).sum(dim=0)

# ❌ DON'T: Use Python loops when vectorization is possible
result = torch.tensor([float(x) ** 2 for x in X])
```

---

## 📚 Additional Resources

- scikit-learn Contributing Guide (for standards inspiration)
- NumPyDoc format reference
- PyTorch documentation

### EXPLANATION

## Key Changes vs scikit-learn

### 1. Backend

- **Replaced**: NumPy/SciPy arrays
- **With**: PyTorch tensors (`torch.Tensor`) and `torch.linalg`

### 2. Maintained Standards

- Kept scikit-learn-style folder structure
- Kept testing + documentation expectations
- Kept estimator API conventions

### 3. Release notes

- Consolidated in a single `RELEASES.md`
- `doc/whats_new/` can remain as a light index that links to `RELEASES.md`

---

## Usage

```bash
# View release notes
cat RELEASES.md

# Add new release notes
echo "## 0.1.1 - 2026-07-10" >> RELEASES.md
```

These are coding guidelines used in torml (scikit-learn-style), with PyTorch as the backend.
