"""Tests for torml.preprocessing."""

from __future__ import annotations

import pytest
import torch

from torml.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from torml.utils import NotFittedError


@pytest.fixture
def numeric_data():
    torch.manual_seed(0)
    return torch.randn(20, 3) * 4 + 5


class TestStandardScaler:
    def test_fit_transform_round_trip(self, numeric_data):
        sc = StandardScaler().fit(numeric_data)
        Xt = sc.transform(numeric_data)
        assert tuple(Xt.shape) == tuple(numeric_data.shape)
        torch.testing.assert_close(Xt.mean(dim=0), torch.zeros(3), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            Xt.std(dim=0, unbiased=False), torch.ones(3), rtol=1e-4, atol=1e-4
        )
        back = sc.inverse_transform(Xt)
        torch.testing.assert_close(back, numeric_data, rtol=1e-4, atol=1e-4)

    def test_fit_transform_method(self, numeric_data):
        Xt = StandardScaler().fit_transform(numeric_data)
        assert tuple(Xt.shape) == (20, 3)

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            StandardScaler().transform(torch.randn(4, 3))

    def test_feature_mismatch_raises(self, numeric_data):
        sc = StandardScaler().fit(numeric_data)
        with pytest.raises(ValueError, match="features"):
            sc.transform(torch.randn(4, 2))

    def test_with_mean_false(self, numeric_data):
        sc = StandardScaler(with_mean=False, with_std=False).fit(numeric_data)
        Xt = sc.transform(numeric_data)
        torch.testing.assert_close(Xt, numeric_data)

    def test_constant_feature(self):
        X = torch.tensor([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        Xt = StandardScaler().fit_transform(X)
        assert torch.isfinite(Xt).all()
        assert float(Xt[:, 0].abs().max()) == 0.0

    def test_get_params(self):
        sc = StandardScaler(with_mean=False)
        assert sc.get_params()["with_mean"] is False
        assert sc.set_params(with_std=False).with_std is False


class TestMinMaxScaler:
    def test_range_and_round_trip(self, numeric_data):
        sc = MinMaxScaler().fit(numeric_data)
        Xt = sc.transform(numeric_data)
        assert float(Xt.min()) >= -1e-6
        assert float(Xt.max()) <= 1.0 + 1e-6
        back = sc.inverse_transform(Xt)
        torch.testing.assert_close(back, numeric_data, rtol=1e-4, atol=1e-4)

    def test_custom_range(self, numeric_data):
        Xt = MinMaxScaler(feature_range=(-1, 1)).fit_transform(numeric_data)
        assert float(Xt.min()) >= -1.0 - 1e-6
        assert float(Xt.max()) <= 1.0 + 1e-6

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            MinMaxScaler().transform(torch.randn(4, 2))

    def test_invalid_range_raises(self, numeric_data):
        with pytest.raises(ValueError, match="min must be < max"):
            MinMaxScaler(feature_range=(1, 0)).fit(numeric_data)

    def test_constant_feature(self):
        X = torch.tensor([[2.0], [2.0], [2.0]])
        Xt = MinMaxScaler().fit_transform(X)
        assert torch.isfinite(Xt).all()


class TestLabelEncoder:
    def test_tensor_round_trip(self):
        y = torch.tensor([2, 0, 1, 2, 0])
        enc = LabelEncoder().fit(y)
        assert enc.classes_.tolist() == [0, 1, 2]
        out = enc.transform(torch.tensor([0, 2, 1]))
        assert out.tolist() == [0, 2, 1]
        back = enc.inverse_transform(out)
        assert back.tolist() == [0, 2, 1]

    def test_string_labels(self):
        enc = LabelEncoder().fit(["b", "a", "b", "c"])
        assert list(enc.classes_) == ["a", "b", "c"]
        assert enc.transform(["a", "c"]).tolist() == [0, 2]
        assert enc.inverse_transform([0, 2]) == ["a", "c"]

    def test_unknown_raises(self):
        enc = LabelEncoder().fit([0, 1])
        with pytest.raises(ValueError, match="Unknown label"):
            enc.transform([5])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            LabelEncoder().fit([])


class TestOneHotEncoder:
    def test_tensor_basic(self):
        X = torch.tensor([[0, 1], [1, 0], [0, 0]])
        enc = OneHotEncoder().fit(X)
        assert enc.n_features_out_ == 4
        Xt = enc.transform(X)
        assert tuple(Xt.shape) == (3, 4)
        assert float(Xt.sum()) == 6.0
        rows = enc.inverse_transform(Xt)
        assert rows[0] == [0, 1]

    def test_string_categories(self):
        X = [["a", "x"], ["b", "x"], ["a", "y"]]
        enc = OneHotEncoder().fit(X)
        assert enc.categories_ == [["a", "b"], ["x", "y"]]
        Xt = enc.transform([["b", "y"]])
        assert tuple(Xt.shape) == (1, 4)
        assert Xt.tolist() == [[0.0, 1.0, 0.0, 1.0]]

    def test_unknown_error_and_ignore(self):
        X = torch.tensor([[0], [1]])
        with pytest.raises(ValueError, match="Unknown category"):
            OneHotEncoder().fit(X).transform(torch.tensor([[5]]))
        Xt = (
            OneHotEncoder(handle_unknown="ignore").fit(X).transform(torch.tensor([[5]]))
        )
        assert Xt.tolist() == [[0.0, 0.0]]

    def test_feature_mismatch_raises(self):
        enc = OneHotEncoder().fit(torch.tensor([[0, 1], [1, 0]]))
        with pytest.raises(ValueError, match="features"):
            enc.transform(torch.tensor([[0]]))

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            OneHotEncoder().transform(torch.tensor([[0]]))
