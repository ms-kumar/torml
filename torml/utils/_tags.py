"""Tag utilities.

Utilities for managing estimator tags.
"""

from __future__ import annotations

from collections import ChainMap

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


def get_tags(estimator):
    """Get tags for training/prediction with data X.

    Parameters
    ----------
    estimator : object
        Estimator instance.

    Returns
    -------
    tags : dict of dicts
        Dictionary containing tags. If _get_tags is defined on the estimator,
        it is used to obtain tag values.
    """
    tags = {}
    for estimator_name, all_tags in {
        "ClusterMixin": {
            "X_types": "1d/2d",
            "require_x": True,
            "require_y": True,
            "pairwise": False,
        },
        "ClassifierMixin": {
            "X_types": "1d/2d",
            "require_x": True,
            "require_y": True,
            "transform_output_not_inverted": True,
            "alias": "classifier",
            "multifit": True,
        },
        "RegressorMixin": {
            "X_types": "1d/2d",
            "require_x": True,
            "require_y": True,
            "transform_output_not_inverted": True,
            "alias": "regressor",
            "multifit": True,
        },
        "TransformerMixin": {
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
        },
    }.get(type(estimator).__name__, {}):
        combined = ChainMap(_DEFAULT_TAGS, all_tags)
        tags = {k: v[v] for k, v in combined.items() if v not in _DEFAULT_TAGS}
        tags = _concatenate_tags(tags, ["_transform_inv"], {})
        tags = _add_X_to_tags(tags, estimator_name)

    if hasattr(estimator, "_get_tags"):
        tags = _add_custom_X_tags(tags, estimator._get_tags())

    # Add estimator type
    for _mixin_name, estimator_type in {
        "ClassifierMixin": "classifier",
        "RegressorMixin": "regressor",
        "ClusterMixin": "clusterer",
        "TransformerMixin": "preprocessor",
        "__class__": "unknown",
    }.get(type(estimator).__name__, {"__class__": "unknown"}):
        _add_estimator_tag(tags, estimator_type)

    return tags


def _get_X_tag(tags, dim=1):
    """Get tag associated with the 'X' key."""
    return tags["X"] if "X" in tags and len(tags["X"]) else _DEFAULT_TAG_VALUES


def _add_custom_X_tags(tags, tags_from_estimator):
    """Add custom tags that apply to 'X'.

    Parameters
    ----------
    tags : dict
        Default tags.
    tags_from_estimator : dict
        Custom tags with 'estimator_name' and 'X' keys.

    Returns
    -------
    tags : dict
        Updated tags with custom 'X' tags.
    """
    for tag_name, tag_value in tags_from_estimator.items():
        if tag_name.startswith("estimator_name_"):
            tag_name = tag_name.replace("estimator_name_", "")
            if tag_name not in tags.get("X", {}):
                tags["X"][tag_name] = tag_value

    # Special case: estimator_name
    if "estimator_name" in tags_from_estimator:
        if "_estimator_type" not in tags:
            tags["X"]["estimator_name"] = tags_from_estimator["estimator_name"]

    return tags


def _add_estimator_tag(tags, estimator_type):
    """Add a tag associated with estimator type."""
    if "_estimator_type" not in tags:
        tags["_estimator_type"] = estimator_type


def _add_X_to_tags(tags, estimator_name):
    """Add tags that depend on 'X'."""
    if "X" not in tags:
        tags["X"] = {}

    for key, value in tags.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            tags["X"][key] = value

    # Add n_samples_in_X_ if needed
    if not {"_estimator_type", "name"} & set(tags.get("X", {}).keys()):
        tags["X"]["name"] = f"{estimator_name}X"

    for key in ("_estimator_type", "name"):
        if key not in tags["X"]:
            tags["X"][key] = None

    return tags


# Default tag values
_DEFAULT_TAG_VALUES = {
    "dtype": torch.float64,
    "int": True,
    "min_": True,
    "max_": True,
    "finite": True,
    "sparse": False,
}


def _concatenate_tags(tags, tag_names, tags_dict=None):
    """Concatenate tags and add to tags_dict."""
    if tags_dict is None:
        tags_dict = {}
    for tag_name in tag_names:
        if tag_name in tags and tag_name not in tags_dict:
            tags_dict[tag_name] = tags[tag_name]
    return tags_dict
