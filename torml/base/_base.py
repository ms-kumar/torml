"""Base estimator implementations.

Core classes for estimator functionality.
"""

from __future__ import annotations

import inspect

import torch

from torml.utils._validation import check_is_fitted


class _MixinABC(type):
    """Base metaclass for mixins to provide lazy-import functionality in score()."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._module = namespace.get("_module", None)


class BaseEstimator:
    """Base class for all estimators.

    Provides common functionality: parameter handling, representation,
    and tags. All estimators should inherit from this class.

    Parameters
    ----------
    *params
        Hyperparameters for the estimator.

    Attributes
    ----------
    _estimator_type : str
        Type of estimator: "classifier", "regressor", or "clusterer".
    name : str
        Name of the estimator.
    """

    _module: str | None = None
    name: str = "Auto"
    _estimator_type: str = "unknown"

    def __init__(self, **kwargs):
        """Initialize the estimator with the given parameters.

        Parameters
        ----------
        **kwargs
            Hyperparameters for the estimator.
        """
        self._base_params = {}

    def get_params(self, deep: bool = True) -> dict[str, any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return deep parameters of estimators.

        Returns
        -------
        dict
            Dictionary of parameter name -> parameter values.
        """
        init_sig = inspect.signature(self.__init__)
        init_params = [p for p in init_sig.parameters if p != "self"]
        params = {}
        for key in init_params:
            if hasattr(self, key):
                value = getattr(self, key)
            else:
                # Fall back to class-level default if instance attr missing.
                default = init_sig.parameters[key].default
                if default is inspect.Parameter.empty:
                    continue
                value = default
            params[key] = value
            if deep and isinstance(value, BaseEstimator):
                nested = value.get_params(deep=True)
                for n_key, n_val in nested.items():
                    params[f"{key}__{n_key}"] = n_val
        return params

    def set_params(self, **params) -> "BaseEstimator":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Parameter names and values to set.

        Returns
        -------
        self : BaseEstimator
            Estimator instance with updated parameters.

        Raises
        ------
        ValueError
            If an unknown parameter is provided.
        """
        known_params = set(inspect.signature(self.__init__).parameters.keys())
        known_params.discard("self")
        known_params.discard("_base_name")

        for key in params:
            if key not in known_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for {self.name!r}. "
                    f"Check the {self.name} documentation for valid parameter values."
                )

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def __repr__(self) -> str:
        """String representation of the object.

        Returns
        -------
        str
            String representation of the object.
        """
        try:
            params = self.get_params(deep=False)
            param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
            return f"{type(self).__name__}({param_str})"
        except Exception:
            return f"<{self.name}>"

    def _more_tags(self) -> dict:
        """Tags for this estimator.

        Override to add estimator-specific tags.

        Returns
        -------
        dict
            Dictionary of tag -> value.
        """
        return {"requirement": "any", "approximation": "no", "preserves_dtype": []}

    def _get_tags(self, X) -> dict:
        """Get tags for training/prediction with data X

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        Returns
        -------
        dict
            Dictionary of tags with 'X' as key.
        """
        tags = self._more_tags()
        tags["X"] = {}

        if hasattr(X, "dtype"):
            tags["X"]["dtype"] = X.dtype
        else:
            try:
                tags["X"]["dtype"] = torch.as_tensor(X).dtype
            except Exception:
                tags["X"]["dtype"] = None

        if hasattr(X, "device"):
            tags["X"]["device"] = X.device

        return tags

    def _check_X(
        self, X, input_name: str | None = None, check_interval=0
    ) -> torch.Tensor:
        """Preprocess X for fitting.

        Check X array for dtype, ensure 2D, allow nd, and copy as needed.
        """
        from torml.utils._validation import check_array

        return check_array(
            X,
            dtype=torch.float32,
            ensure_2d=True,
            allow_nd=False,
            copy=True,
            force_all_finite=True,
        )

    def _check_y(
        self, y, input_name: str | None = None, check_interval=0
    ) -> torch.Tensor:
        """Preprocess y for fitting.

        Check y array for dtype, ensure 1D, allow nd=False, and copy as needed.
        """
        from torml.utils._validation import check_array

        return check_array(
            y,
            dtype=torch.float32,
            ensure_2d=False,
            allow_nd=False,
            copy=True,
            force_all_finite=True,
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> "BaseEstimator":
        """Compute all parameters of the estimator.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training vectors.
        y : torch.Tensor of shape (n_samples,), optional
            Target values.

        Returns
        -------
        self : BaseEstimator
            Fitted estimator with attributes set. Returns self for chaining.
        """
        raise NotImplementedError(f"fit not implemented for {self.name}")

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform X using this estimator.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        X_new : torch.Tensor of shape (n_samples, n_transformed_features)
            Transformed samples.

        Raises
        ------
        NotFittedError
            If the estimator is not fitted.
        """
        check_is_fitted(self)
        return self._transform(X)

    def fit_transform(
        self, X: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fit to X, then transform X.

        Equivalent to `transform(self.fit(X, y))`.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training vectors.
        y : torch.Tensor of shape (n_samples,), optional
            Target values.

        Returns
        -------
        X_transformed : torch.Tensor
            Transformed samples.

        Notes
        -----
        The returned data is equal to transform(fitted_estimator) but fit_transform
        is more efficient as the transformation is done only once.
        """
        fit_return = self.fit(X, y)
        return fit_return.transform(X)

    def score(
        self, X: torch.Tensor, y: torch.Tensor | None = None
    ) -> float | torch.Tensor:
        """Score predictions against targets without refitting.

        Parameters
        ----------
        X : torch.Tensor
            Input samples.
        y : torch.Tensor, optional
            Target values.

        Returns
        -------
        score : float or torch.Tensor
            The model score (R², accuracy, etc).
        """
        return self._score(X, y)

    def _score(
        self, X: torch.Tensor, y: torch.Tensor | None = None
    ) -> float | torch.Tensor:
        """Compute score. Override to implement custom score, calls base."""
        raise NotImplementedError


class ClassifierMixin(BaseEstimator):
    """Mixin for classifiers.

    Provides a default `score` method that calls accuracy_score.
    """

    _estimator_type = "classifier"

    def _score(self, X: torch.Tensor, y: torch.Tensor | None = None) -> float:
        from torml.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class RegressorMixin(BaseEstimator):
    """Mixin for regressors.

    Provides a default `score` method that calls r2_score.
    """

    _estimator_type = "regressor"

    def _score(self, X: torch.Tensor, y: torch.Tensor | None = None) -> float:
        from torml.metrics import r2_score

        y_pred = self.predict(X)
        if y is None:
            raise ValueError("y must be provided for regression score")
        return r2_score(y, y_pred)


class TransformerMixin(BaseEstimator):
    """Mixin for transformers.

    Provides a default `fit_transform` implementation.
    """

    _estimator_type = "preprocessor"
    n_features_out_: int = 0

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform X. Override to implement custom transform."""
        raise NotImplementedError

    def fit_transform(
        self, X: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fit to X, then transform X.

        Override fit_transform only if you cannot express your transform
        as fit(X).y, transform(X).
        """
        fitted = self.fit(X)
        return fitted.transform(X)


class ClusterMixin(BaseEstimator):
    """Mixin for clusterers.

    Provides a default `fit_predict` implementation.
    """

    _estimator_type = "clusterer"

    def _fit(self, X: torch.Tensor) -> "ClusterMixin":
        """Fit the model. Override to implement custom fit."""
        raise NotImplementedError

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and return labels.

        Override fit() only if you cannot express your prediction
        as fit(X).labels.
        """
        self.fit(X)
        return self.labels_


def clone(estimator, *, safe: bool = True) -> "BaseEstimator":
    """Create a newly fitted estimator over the original one.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator instance to clone.
    safe : bool, default=True
        If True, raise TypeError if estimator is not an instance of BaseEstimator.

    Returns
    -------
    clone : BaseEstimator
        A new estimator with the same parameters as the original.

    Raises
    ------
    TypeError
        If safe=True and estimator is not a BaseEstimator instance.
    """
    if not isinstance(estimator, BaseEstimator):
        if safe:
            raise TypeError("Unable to clone non-BaseEstimator.")
        import copy

        return copy.deepcopy(estimator)

    # Get params to include all subclass params, excluding nested __ params.
    params = estimator.get_params(deep=False)

    # Use the estimator's __init__ signature to know what params it accepts
    sig_params = set(inspect.signature(estimator.__class__.__init__).parameters.keys())
    sig_params.discard("self")

    # Filter params to only include those the init accepts
    filtered_params = {k: v for k, v in params.items() if k in sig_params}

    return estimator.__class__(**filtered_params)


def is_classifier(estimator) -> bool:
    """Check if estimator is a classifier.

    Parameters
    ----------
    estimator : object
        Estimator instance or class.

    Returns
    -------
    bool
        True if estimator is a classifier, False otherwise.
    """
    if isinstance(estimator, type):  # Check for class
        return issubclass(estimator, ClassifierMixin)
    return isinstance(estimator, ClassifierMixin)


def is_regressor(estimator) -> bool:
    """Check if estimator is a regressor.

    Parameters
    ----------
    estimator : object
        Estimator instance or class.

    Returns
    -------
    bool
        True if estimator is a regressor, False otherwise.
    """
    if isinstance(estimator, type):  # Check for class
        return issubclass(estimator, RegressorMixin)
    return isinstance(estimator, RegressorMixin)


def is_clusterer(estimator) -> bool:
    """Check if estimator is a clusterer.

    Parameters
    ----------
    estimator : object
        Estimator instance or class.

    Returns
    -------
    bool
        True if estimator is a clusterer, False otherwise.
    """
    if isinstance(estimator, type):  # Check for class
        return issubclass(estimator, ClusterMixin)
    return isinstance(estimator, ClusterMixin)


class CloneMixin:
    """CloneMixin for cloning without deep copy."""

    def __init__(self):
        pass

    def clone(self, deep=False):
        """Clone this object.

        Parameters
        ----------
        deep : bool, default=False
            If True, use deep copy.

        Returns
        -------
        CloneMixin
            A cloned instance.
        """
        import copy

        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def copy(self, deep=False):
        """Copy this object.

        Parameters
        ----------
        deep : bool, default=False
            If True, use deep copy.

        Returns
        -------
        A copy instance.
        """
        import copy as deepcopy

        if deep:
            return deepcopy.deepcopy(self)
        else:
            # Create a shallow copy that copies __dict__ attributes
            result = type(self)()
            result.__dict__.update(self.__dict__)
            return result
