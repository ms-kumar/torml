---
name: validation-and-testing
description: 'Use when: adding torml input validation, check_array, check_X_y, check_is_fitted, pytest tests, torch.testing assertions, estimator checks, shape validation, not-fitted behavior.'
argument-hint: '<module or estimator>'
---

# Validation and Testing

Use this skill when creating shared validation helpers or tests.

## Validation Procedure

1. Prefer shared helpers in `torml/utils/_validation.py`.
2. Validate shape, dtype, dimensionality, finite values, and sample count.
3. Store `n_features_in_` during `fit` and check it in later calls.
4. Raise `ValueError` for invalid values/shapes and `TypeError` for invalid parameter types.

## Test Procedure

1. Put tests under `tests/<module>/` or matching top-level test file.
2. Use deterministic small tensors.
3. Test happy path, invalid input, and not-fitted errors.
4. Use `torch.testing.assert_close` for numerical comparisons.
5. Use `pytest.mark.parametrize` for hyperparameter variants.

## Minimum Test Checklist

- Fit returns `self`.
- Predict/transform output shape is correct.
- Invalid `X`/`y` shapes fail.
- Calling predict/transform before fit fails.
- Learned attributes exist after fit.
