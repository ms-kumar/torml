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
  host round-trips, no device-mismatch errors. Random generators are created
  on the data device, and class tables live next to the data they index.
- Apple Silicon (MPS) is supported the same way and is exercised by gated
  tests wherever MPS exists (``tests/common/test_dtype_device.py``).
- CPU remains the fully tested path: the suite runs on CPU, and CUDA/MPS
  coverage is ``requires CUDA``/``requires MPS``-gated.

.. code-block:: python

   X = torch.randn(40, 3, device="cuda")   # or "mps"
   StandardScaler().fit_transform(X).device.type   # 'cuda'

Caveat: on Apple Silicon, a few ``torch.linalg`` ops historically lagged on
MPS — if you hit ``NotImplementedError`` from inside ``torch.linalg``, move
that step to CPU. Majority votes use a gather-based implementation instead
of ``torch.mode`` for exactly this reason.
