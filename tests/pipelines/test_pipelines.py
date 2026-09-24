"""Tests for torml.pipelines."""

from __future__ import annotations

import pytest
import torch

from torml.linear_model import LinearRegression
from torml.neighbors import KNeighborsClassifier
from torml.pipelines import ColumnTransformer, FeatureUnion, Pipeline
from torml.preprocessing import MinMaxScaler, StandardScaler
from torml.utils import NotFittedError


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(30, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(30, 2) + torch.tensor([3.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    return X, y


class TestPipeline:
    def test_scale_then_classify(self, blobs):
        X, y = blobs
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(3))]
        ).fit(X, y)
        assert float((pipe.predict(X) == y).float().mean()) > 0.95
        assert float(pipe.score(X, y)) > 0.95

    def test_regression(self):
        torch.manual_seed(1)
        X = torch.randn(50, 2)
        y = 2 * X[:, 0] - X[:, 1]
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("reg", LinearRegression())]
        ).fit(X, y)
        assert float(pipe.score(X, y)) > 0.99

    def test_nested_params(self, blobs):
        X, y = blobs
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(1))]
        )
        assert pipe.get_params()["clf__n_neighbors"] == 1
        pipe.set_params(clf__n_neighbors=5)
        assert pipe.get_params()["clf__n_neighbors"] == 5
        pipe.fit(X, y)

    def test_transform_only(self, blobs):
        X, _ = blobs
        pipe = Pipeline([("scaler", StandardScaler()), ("minmax", MinMaxScaler())]).fit(
            X
        )
        assert tuple(pipe.transform(X).shape) == (60, 2)

    def test_not_fitted_raises(self):
        with pytest.raises((NotFittedError, ValueError)):
            Pipeline([("scaler", StandardScaler())]).predict(torch.randn(2, 2))

    def test_duplicate_names_raise(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Pipeline([("a", StandardScaler()), ("a", MinMaxScaler())]).fit(
                torch.randn(5, 2)
            )


class TestUnion:
    def test_feature_union(self, blobs):
        X, _ = blobs
        union = FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())]).fit(X)
        assert tuple(union.transform(X).shape) == (60, 4)

    def test_column_transformer(self, blobs):
        X, _ = blobs
        ct = ColumnTransformer(
            [("std", StandardScaler(), [0]), ("mm", MinMaxScaler(), [1])]
        ).fit(X)
        assert tuple(ct.transform(X).shape) == (60, 2)

    def test_remainder_passthrough(self, blobs):
        X, _ = blobs
        ct = ColumnTransformer(
            [("std", StandardScaler(), [0])], remainder="passthrough"
        ).fit(X)
        assert tuple(ct.transform(X).shape) == (60, 2)

    def test_bad_cols_raise(self, blobs):
        X, _ = blobs
        with pytest.raises(ValueError, match="out of range"):
            ColumnTransformer([("std", StandardScaler(), [7])]).fit(X)
