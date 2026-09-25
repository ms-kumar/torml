"""Tests for torml.utils.get_tags."""

from __future__ import annotations

import torch

from torml.base import BaseEstimator
from torml.cluster import KMeans
from torml.linear_model import LinearRegression, LogisticRegression
from torml.preprocessing import StandardScaler
from torml.utils import get_tags
from torml.utils._tags import _DEFAULT_TAGS


class TestGetTags:
    def test_classifier(self):
        tags = get_tags(LogisticRegression())
        assert tags["_estimator_type"] == "classifier"
        assert tags["require_y"] is True
        assert tags["alias"] == "classifier"

    def test_regressor(self):
        tags = get_tags(LinearRegression())
        assert tags["_estimator_type"] == "regressor"
        assert tags["require_y"] is True
        assert tags["alias"] == "regressor"

    def test_clusterer(self):
        tags = get_tags(KMeans(n_clusters=2))
        assert tags["_estimator_type"] == "clusterer"
        assert tags["pairwise"] is False

    def test_transformer(self):
        tags = get_tags(StandardScaler())
        assert tags["_estimator_type"] == "preprocessor"
        assert tags["require_y"] is False

    def test_unknown_estimator(self):
        tags = get_tags(object())
        assert tags["_estimator_type"] == "unknown"
        for key, value in _DEFAULT_TAGS.items():
            assert tags[key] == value

    def test_plain_base_estimator(self):
        tags = get_tags(BaseEstimator())
        assert tags["_estimator_type"] == "unknown"

    def test_does_not_mutate_defaults(self):
        before = dict(_DEFAULT_TAGS)
        get_tags(LogisticRegression())
        assert _DEFAULT_TAGS == before

    def test_torch_free_input(self):
        tags = get_tags(torch.nn.Module())
        assert tags["_estimator_type"] == "unknown"
