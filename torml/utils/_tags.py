"""Tag utilities.

Utilities for managing estimator tags.
"""

from __future__ import annotations

import torch

_DEFAULT_TAGS = {
    "X_types": "1d/2d",
    "require_x": True,
    "require_y": False,
    "transform_output_not_inverted": True,
    "pairwise": True,
    "alias": None,
    "multifit": False,
    "no_nan": True,
    "allow_infinite": False,
    "sparse_input": False,
    "assume_is_fitted": ["_label_indices"],
}

_CLASSIFIER_TAGS = {
    "X_types": "1d/2d",
    "require_x": True,
    "require_y": True,
    "transform_output_not_inverted": True,
    "alias": "classifier",
    "multifit": True,
}

_REGRESSOR_TAGS = {
    "X_types": "1d/2d",
    "require_x": True,
    "require_y": True,
    "transform_output_not_inverted": True,
    "alias": "regressor",
    "multifit": True,
}

_CLUSTER_TAGS = {
    "X_types": "1d/2d",
    "require_x": True,
    "require_y": True,
    "pairwise": False,
}

_TRANSFORMER_TAGS = {
    "X_types": "1d/2d",
    "require_x": True,
    "require_y": False,
    "_transform_inv": False,
    "transform_output_not_inverted": False,
    "alias": "preprocessor",
    "_skip_check_finite": True,
    "_skip_check_nan": True,
    "_skip_check_min_samples": True,
    "_skip_check_min_features": True,
    "assume_is_fitted": ["_transform_weights"],
}


def get_tags(estimator):
    """Get tags describing an estimator's requirements.

    Tags are resolved with :func:`isinstance` against the mixin types,
    so subclasses inherit their parents' tags. Priority for
    ``_estimator_type`` is classifier, regressor, clusterer, transformer.

    Parameters
    ----------
    estimator : object
        Estimator instance.

    Returns
    -------
    tags : dict
        Defaults from ``_DEFAULT_TAGS`` overridden by the estimator type,
        plus ``_estimator_type`` (``"unknown"`` for non-estimators).
    """
    from torml.base import (
        ClassifierMixin,
        ClusterMixin,
        RegressorMixin,
        TransformerMixin,
    )

    tags = dict(_DEFAULT_TAGS)
    estimator_type = "unknown"
    if isinstance(estimator, ClassifierMixin):
        tags.update(_CLASSIFIER_TAGS)
        estimator_type = "classifier"
    elif isinstance(estimator, RegressorMixin):
        tags.update(_REGRESSOR_TAGS)
        estimator_type = "regressor"
    elif isinstance(estimator, ClusterMixin):
        tags.update(_CLUSTER_TAGS)
        estimator_type = "clusterer"
    elif isinstance(estimator, TransformerMixin):
        tags.update(_TRANSFORMER_TAGS)
        estimator_type = "preprocessor"
    tags["_estimator_type"] = estimator_type
    return tags


def _get_X_tag(tags, dim=1):
    """Get tag associated with the 'X' key."""
    return tags["X"] if "X" in tags and len(tags["X"]) else _DEFAULT_TAG_VALUES


# Default tag values
_DEFAULT_TAG_VALUES = {
    "dtype": torch.float64,
    "int": True,
    "min_": True,
    "max_": True,
    "finite": True,
    "sparse": False,
}
