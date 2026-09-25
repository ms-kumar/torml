"""Estimator checks.

Generic testing utilities that check if estimators follow conventions.
"""

from __future__ import annotations


def check_estimator(estimator):
    """Test that estimators conform to API conventions.

    Parameters
    ----------
    estimator : object
        Estimator instance to check.

    Checks
    -------
    - get_params/set_params round-trip
    - repr doesn't raise
    - estimator has a class name
    - clone returns a fitted-capable copy
    """

    # Test get_params/set_params round-trip (repr-normalized: meta-estimators
    # hold live sub-estimators, and fresh clones never compare identical)
    original_params = estimator.get_params()
    reconstructed = estimator.set_params(**original_params)
    new_params = reconstructed.get_params()

    def _normalize(params):
        return {key: repr(value) for key, value in params.items()}

    if _normalize(original_params) != _normalize(new_params):
        raise AssertionError("get_params/set_params failed")

    # Test repr doesn't raise or return type
    try:
        repr_str = repr(estimator)
        if not isinstance(repr_str, str):
            raise AssertionError("repr must return string")
        if type(estimator).__name__ not in repr_str:
            raise AssertionError("repr missing class name")
    except Exception as e:
        raise AssertionError(f"repr failed: {e}") from e

    # Test estimator has a usable class name (no suffix rule: sklearn's own
    # LinearRegression/LogisticRegression don't end in Regressor/Classifier)
    name = type(estimator).__name__
    if not name:
        raise AssertionError("Estimator failed validation. No name")

    # Test clone
    from torml.base import clone

    cloned_estimator = clone(estimator)
    if cloned_estimator is estimator:
        raise AssertionError("Clone returned original")
    if not (hasattr(cloned_estimator, "fit") and hasattr(cloned_estimator, "predict")):
        raise AssertionError("Clone missing methods")

    return estimator
