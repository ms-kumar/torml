"""Model selection utilities.

Provides data splitting and cross-validation with a PyTorch backend.
"""

from __future__ import annotations

from ._split import KFold, train_test_split
from ._validation import cross_val_score

__all__ = ["KFold", "cross_val_score", "train_test_split"]
