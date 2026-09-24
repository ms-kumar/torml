"""Tests for torml.model_selection."""

from __future__ import annotations

import pytest
import torch

from torml.linear_model import LinearRegression
from torml.model_selection import KFold, cross_val_score, train_test_split


@pytest.fixture
def regression_data():
    torch.manual_seed(0)
    X = torch.randn(40, 2)
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5
    return X, y


class TestTrainTestSplit:
    def test_default_sizes(self, regression_data):
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        assert X_train.shape[0] == 30
        assert X_test.shape[0] == 10
        assert y_train.shape[0] == 30
        assert y_test.shape[0] == 10

    def test_no_shuffle_is_deterministic(self, regression_data):
        X, y = regression_data
        X_train, X_test, _, _ = train_test_split(X, y, test_size=8, shuffle=False)
        assert torch.equal(X_train, X[:32])
        assert torch.equal(X_test, X[32:])

    def test_random_state_reproducible(self, regression_data):
        X, y = regression_data
        out1 = train_test_split(X, y, test_size=10, random_state=42)
        out2 = train_test_split(X, y, test_size=10, random_state=42)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])

    def test_mismatched_lengths_raise(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="same number of samples"):
            train_test_split(X, y[:10])

    def test_invalid_sizes_raise(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, y, test_size=100)
        with pytest.raises(ValueError, match="train_size.*test_size|must be <="):
            train_test_split(X, y, train_size=35, test_size=10)

    def test_stratify_not_supported(self, regression_data):
        X, y = regression_data
        with pytest.raises(NotImplementedError):
            train_test_split(X, y, stratify=y)


class TestKFold:
    def test_covers_all_samples_no_overlap(self):
        X = torch.randn(10, 3)
        kf = KFold(n_splits=5)
        seen = torch.zeros(10, dtype=torch.bool)
        for train_idx, test_idx in kf.split(X):
            assert len(test_idx) == 2
            assert len(train_idx) == 8
            assert not torch.isin(test_idx, train_idx).any()
            seen[test_idx] = True
        assert seen.all()

    def test_get_n_splits(self):
        assert KFold(n_splits=4).get_n_splits() == 4

    def test_get_params_round_trip(self):
        kf = KFold(n_splits=3, shuffle=True, random_state=1)
        params = kf.get_params()
        assert params["n_splits"] == 3
        assert kf.set_params(n_splits=4).n_splits == 4

    def test_shuffle_reproducible(self):
        X = torch.arange(12).unsqueeze(1).float()
        splits1 = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X))
        splits2 = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X))
        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            assert torch.equal(tr1, tr2)
            assert torch.equal(te1, te2)

    def test_invalid_n_splits(self):
        X = torch.randn(8, 2)
        with pytest.raises(ValueError, match="at least 2"):
            list(KFold(n_splits=1).split(X))
        with pytest.raises(ValueError, match="n_samples"):
            list(KFold(n_splits=10).split(X))

    def test_random_state_without_shuffle_raises(self):
        X = torch.randn(8, 2)
        with pytest.raises(ValueError, match="shuffle=False"):
            list(KFold(n_splits=2, shuffle=False, random_state=0).split(X))


class TestCrossValScore:
    def test_returns_n_scores_and_high_r2(self, regression_data):
        X, y = regression_data
        scores = cross_val_score(LinearRegression(), X, y, cv=4)
        assert isinstance(scores, torch.Tensor)
        assert tuple(scores.shape) == (4,)
        assert bool((scores > 0.99).all())

    def test_cv_splitter_object(self, regression_data):
        X, y = regression_data
        scores = cross_val_score(
            LinearRegression(), X, y, cv=KFold(n_splits=2, shuffle=False)
        )
        assert tuple(scores.shape) == (2,)

    def test_scoring_string(self, regression_data):
        X, y = regression_data
        scores = cross_val_score(LinearRegression(), X, y, cv=2, scoring="r2")
        assert tuple(scores.shape) == (2,)

    def test_invalid_cv_raises(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="at least 2"):
            cross_val_score(LinearRegression(), X, y, cv=1)
        with pytest.raises(TypeError, match="cv must be"):
            cross_val_score(LinearRegression(), X, y, cv="bad")

    def test_mismatched_X_y_raises(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="inconsistent lengths"):
            cross_val_score(LinearRegression(), X, y[:5], cv=2)
