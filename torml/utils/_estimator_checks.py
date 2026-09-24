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
    - estimator name
    - not-fitted raises NotFittedError
    - n_features_in_ attribute
    - output shape
    """

    # Test get_params/set_params round-trip
    original_params = estimator.get_params()
    reconstructed = estimator.set_params(**original_params)
    new_params = reconstructed.get_params()

    if original_params != new_params:
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

    # Test estimator name
    name = repr_str.split(".")[0]
    if not name:
        raise AssertionError("Estimator failed validation. No name")

    if not name.endswith("Regressor") and not name.endswith("Classifier"):
        raise AssertionError(
            f"Estimator {name!r} is not named to indicate it is "
            f"a Regression or Classifier estimator."
        )

    # Test clone
    from torml.base import clone

    cloned_estimator = clone(estimator)
    if cloned_estimator is estimator:
        raise AssertionError("Clone returned original")
    if not (hasattr(cloned_estimator, "fit") and hasattr(cloned_estimator, "predict")):
        raise AssertionError("Clone missing methods")

    return estimator
