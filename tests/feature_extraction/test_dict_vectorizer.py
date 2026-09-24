"""Tests for torml.feature_extraction."""

from __future__ import annotations

import pytest

from torml.feature_extraction import DictVectorizer
from torml.utils import NotFittedError


@pytest.fixture
def rows():
    return [{"a": 1, "b": 2}, {"b": 3, "c": 4}]


class TestDictVectorizer:
    def test_sorted_vocab(self, rows):
        vec = DictVectorizer().fit(rows)
        assert vec.feature_names_ == ["a", "b", "c"]
        Xt = vec.transform(rows)
        assert tuple(Xt.shape) == (2, 3)
        assert Xt[0].tolist() == [1.0, 2.0, 0.0]

    def test_round_trip(self, rows):
        vec = DictVectorizer().fit(rows)
        assert vec.inverse_transform(vec.transform(rows)) == rows

    def test_unknown_raises(self, rows):
        vec = DictVectorizer().fit(rows)
        with pytest.raises(ValueError, match="Unknown feature"):
            vec.transform([{"zzz": 1}])

    def test_not_fitted(self):
        with pytest.raises((NotFittedError, ValueError)):
            DictVectorizer().transform([{"a": 1}])

    def test_names_out(self, rows):
        assert DictVectorizer().fit(rows).get_feature_names_out() == ["a", "b", "c"]
