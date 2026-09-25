Dtypes and devices
====================

torml preserves your input dtype and device instead of silently converting
to ``float32``/CPU.

Dtypes
------

- ``float32`` and ``float64`` tensors keep their precision end to end:
  statistics (``mean_``, ``coef_``, ``components_``) and predictions come
  back in the input dtype.
- Integer feature tensors are promoted to ``float32`` (means and variances
  need fractions); integer *label* tensors stay integer.
- Lists default to ``float32``. Pass ``dtype=`` explicitly to
  ``check_array`` to force a conversion.

.. code-block:: python

   X64 = torch.randn(20, 3, dtype=torch.float64)
   model = LinearRegression().fit(X64, y64)
   model.coef_.dtype   # torch.float64

Devices
-------

- CUDA inputs stay on CUDA through ``fit``/``predict``/``transform`` — no
  host round-trips, no device-mismatch errors.
- CPU remains the tested path: the suite runs on CPU, and CUDA coverage is
  a ``requires CUDA``-gated round-trip test (see
  ``tests/common/test_dtype_device.py``).

.. code-block:: python

   X = torch.randn(40, 3, device="cuda")
   StandardScaler().fit_transform(X).device.type   # 'cuda'

Caveat: on Apple Silicon, some ``torch.linalg`` ops used internally
(``lstsq``, ``svd``, ``cholesky``) have limited MPS support — CPU is the
supported backend there.
