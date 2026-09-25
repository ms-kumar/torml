"""Test base estimator module."""

from __future__ import annotations

import pytest
import torch

from torml.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from torml.utils import NotFittedError, check_is_fitted


class DummyClassifier(ClassifierMixin):
    """Dummy classifier for testing."""

    name = "DummyClassifier"
    _estimator_type = "classifier"

    def __init__(
        self,
        criterion="ce",
        loss=None,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
    ):
        self.criterion = criterion
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, X, y):
        raise NotImplementedError


class DummyRegressor(RegressorMixin):
    """Dummy regressor for testing."""

    name = "DummyRegressor"
    _estimator_type = "regressor"

    def __init__(self, criterion="leak"):
        self.criterion = criterion

    def fit(self, X, y):
        raise NotImplementedError


class _FitDummy(BaseEstimator):
    """Minimal working estimator: fit records features and returns self."""

    def fit(self, X, y=None):
        self.n_features_in_ = int(torch.as_tensor(X).shape[1])
        return self


@pytest.fixture
def dummy_classifier():
    return DummyClassifier()


@pytest.fixture
def dummy_regressor():
    return DummyRegressor()


@pytest.fixture
def X():
    return torch.randn(100, 5)


@pytest.fixture
def y():
    return torch.randn(100)


class TestBaseEstimator:
    """Tests for BaseEstimator."""

    def test_init_hyperparam_names(self):
        """Test that hyperparam names are stored but not learned attributes."""
        estimator = DummyClassifier()

        assert "_base_param" not in dir(estimator)
        assert "_base_param2" not in dir(estimator)
        assert "criterion" in dir(estimator)
        assert "loss" in dir(estimator)
        assert "max_iter" in dir(estimator)
        assert "tol" in dir(estimator)
        assert "warm_start" in dir(estimator)

    def test_set_params_raises(self, dummy_classifier):
        """Test set_params raises ValueError for unknown keys."""
        with pytest.raises(ValueError, match="unknown_key"):
            dummy_classifier.set_params(unknown_key="value")

    def test_set_params_round_trip(self, dummy_classifier):
        """Test set_params returns self and stores values."""
        out = dummy_classifier.set_params(criterion="gini")
        assert out is dummy_classifier
        assert dummy_classifier.criterion == "gini"

    def test_repr(self, dummy_classifier):
        """Test repr doesn't raise."""
        rep_str = repr(dummy_classifier)
        assert rep_str is not None
        assert "DummyClassifier" in rep_str

    def test_not_fitted_raises(self, dummy_classifier):
        """Test check_is_fitted raises NotFittedError before fit."""
        with pytest.raises(NotFittedError):
            check_is_fitted(dummy_classifier)

    def test_fit_returns_self(self, X, y):
        """Test fit returns self."""
        est = _FitDummy()
        assert est.fit(X, y) is est

    def test_n_features_in_set(self, X, y):
        """Test n_features_in_ is set by fit."""
        est = _FitDummy().fit(X, y)
        assert est.n_features_in_ == 5

    def test_is_classifier(self, dummy_classifier):
        """Test is_classifier returns True."""
        assert is_classifier(dummy_classifier) is True

    def test_is_not_classifier(self, dummy_regressor):
        """Test is_classifier returns False for non-classifier."""
        assert is_classifier(dummy_regressor) is False

    def test_is_regressor(self, dummy_regressor):
        """Test is_regressor returns True."""
        assert is_regressor(dummy_regressor) is True

    def test_is_not_regressor(self, dummy_classifier):
        """Test is_regressor returns False for non-regressor."""
        assert is_regressor(dummy_classifier) is False

    def test_clone(self, dummy_classifier):
        """Test clone works."""
        clone_estimator = clone(dummy_classifier)
        assert isinstance(clone_estimator, type(dummy_classifier))
        assert clone_estimator.name == dummy_classifier.name
        assert clone_estimator._estimator_type == dummy_classifier._estimator_type


class TestCloning:
    """Tests for clone function."""

    @pytest.fixture
    def custom_estimator(self):
        class CustomEstimator(BaseEstimator):
            name = "Custom"
            _estimator_type = "unknown"

        return CustomEstimator()

    @pytest.fixture
    def custom_estimator_with_params(self):
        class CustomEstimator2(BaseEstimator):
            name = "Custom2"
            _estimator_type = "unknown"

            def __init__(self, param1="test", param2="test2"):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        return CustomEstimator2(param1="value1", param2="value2")

    def test_clone_returns_different_instance(self, dummy_classifier):
        """Test that clone returns different instance."""
        cloned = clone(dummy_classifier, safe=True)
        assert cloned is not dummy_classifier
        assert isinstance(cloned, type(dummy_classifier))

    def test_unsafe_clone_non_estimator(self):
        """Test that unsafe clone deep-copies non-estimator."""
        original = {"key": "value"}
        cloned = clone(original, safe=False)
        assert cloned == original
        assert cloned is not original

    def test_clone_raises_on_non_estimator(self):
        """Test that safe clone raises on non-estimator."""
        unsafe_estimator = {"key": "value"}
        with pytest.raises(TypeError):
            clone(unsafe_estimator, safe=True)

    def test_clone_raises_on_non_estimator_safe(self):
        """Test that safe clone raises on non-estimator."""
        with pytest.raises(TypeError):
            clone([1, 2, 3], safe=True)

    def test_clone_preserves_params(self, dummy_classifier):
        """Test that clone preserves original params."""
        dummy_classifier.set_params(criterion="test_criterion")
        cloned = clone(dummy_classifier)
        assert cloned.criterion == "test_criterion"

    def test_clone_preserves_init(self):
        """Test that clone preserves init."""
        init_params = DummyClassifier(criterion="ce", loss=None).get_params(deep=False)
        copy = DummyClassifier(**init_params)
        assert "criterion" in copy.get_params(deep=False)

    def test_clone_returns_correct_params_dict(self, dummy_classifier):
        """Test that clone returns correct params dict."""
        cloned = clone(dummy_classifier)
        clone_params = cloned.get_params(deep=False)
        assert "criterion" in clone_params.keys()
        assert isinstance(clone_params["criterion"], str)
        assert clone_params["loss"] is None

    def test_clone_uses_init_params_in_get_params(self):
        """Test that clone uses init params in get_params for base estimators."""
        init_params = DummyClassifier().get_params(deep=False)
        assert "criterion" in init_params
        cloned = clone(DummyClassifier(criterion="new_param"))
        assert cloned.criterion == "new_param"

    def test_clone_correct_params(self, dummy_classifier):
        """Test that clone has correct params from init."""
        init_params = dummy_classifier.get_params(deep=False)
        assert dummy_classifier.criterion == init_params["criterion"]

    def test_clone_returns_correct_params(self, dummy_classifier):
        """Test that clone returns correct params from init."""
        init_params = dummy_classifier.get_params(deep=False)
        cloned = clone(dummy_classifier)
        assert cloned.get_params(deep=False) == init_params
