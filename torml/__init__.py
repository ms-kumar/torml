"""torml: PyTorch-based scikit-learn-style machine learning library.

torml provides a scikit-learn-style API for machine learning in Python,
using PyTorch as the numerical backend.

Available estimators and tools:
- Linear Regression, Logistic Regression
- Clustering algorithms
- Model selection tools

Parameters are stored with `_` prefix (e.g., `coef_`, `labels_`, `intercept_`)
and learned attributes follow scikit-learn naming conventions.

"""

from importlib.metadata import version

__version__ = "0.0.0"
