torml vs scikit-learn
======================

torml mirrors the scikit-learn API on a PyTorch backend. This page reports a
fair, reproducible head-to-head so you know exactly where each library wins.
Rerun it any time:

.. code-block:: bash

   uv run --with scikit-learn python benchmarks/compare_sklearn.py

Method: same data, same seeds, CPU, best of 3 timings, 10k samples per task
(Apple M-series, 2026-09-25, torch 2.14.0, scikit-learn 1.9.1).

Results
-------

=================== ======================= ================== ==================
task                metric                  torml              scikit-learn
=================== ======================= ================== ==================
linreg fit          seconds (best of 3)     0.0012             0.0014
linreg              R²                      0.999656           0.999656
kmeans fit          seconds (best of 3)     0.1631             0.1032
kmeans              inertia (lower better)  72645.4            72611.7
knn fit             seconds (best of 3)     0.0003             0.0018
knn predict-10k     seconds                 5.1368             0.2828
knn                 accuracy                0.9476             0.9476
tree fit            seconds (best of 3)     0.0083             0.0057
tree                accuracy                1.0000             1.0000
scaler              seconds (best of 3)     0.0007             0.0008
scaler              max |diff| vs sklearn   4.77e-07           reference
torch-native        backward() thru predict True               n/a (numpy out)
=================== ======================= ================== ==================

Reading the table honestly: correctness matches everywhere (identical R²,
accuracy, and scaler outputs; kmeans inertia differs only by random
initialization). Timings are in the same class except tree fitting, where
scikit-learn's Cython is ~1.5x faster than torml's Python CART loops, and
kNN prediction at 10k, where scikit-learn's ball tree (~0.3s) beats torml's
exact brute-force pairwise distances (~5s).

At 50k samples with an MPS column
---------------------------------

``benchmarks/compare_500k.py`` reruns the suite at 50k samples per task on
torml-CPU, torml-MPS, and sklearn-CPU (kNN predicts 2k queries):

=================== ======================= ========= ======== ===========
task                metric                  torml-cpu torml-mps sklearn-cpu
=================== ======================= ========= ======== ===========
linreg fit          seconds (best of 3)     0.008     0.015    0.010
linreg              R²                      0.999662  0.999662 0.999662
kmeans fit          seconds (best of 3)     1.977     3.384    0.269
kmeans              inertia (lower better)  366949.9  366817.4 367078.0
knn fit-50k         seconds (best of 3)     0.001     -        0.020
knn predict-2k      seconds                 0.930     0.760    0.260
knn                 accuracy                0.9160    0.9160   0.9160
tree fit-50k        seconds (best of 3)     0.072     0.158    0.059
tree                accuracy                1.0000    1.0000   1.0000
scaler              seconds (best of 3)     0.003     0.004    0.006
=================== ======================= ========= ======== ===========

Takeaways: correctness still ties; MPS helps kNN prediction slightly but
kernel-launch overhead dominates the smaller ops at this scale (and large
``linalg`` calls fall back to CPU with a warning). sklearn's ball tree
(~0.26s) beats exact brute-force kNN predict (~0.9s) — the price of exact
neighbors without an index structure.

Where torml is genuinely better
-------------------------------

- **Torch-native I/O.** Inputs and outputs are ``torch.Tensor`` — zero-copy
  interop with torch pipelines, and predictions stay differentiable
  (``backward()`` flows through ``LinearRegression.predict``).
- **One runtime dependency** (``torch``) instead of a NumPy/SciPy stack.
- **Exact statistics** where sklearn approximates — e.g. ``f_classif``
  p-values are computed exactly, not via approximations.
- **Readable from-scratch code** with strict input validation on every
  estimator; useful for teaching and auditing.
- **Dtypes and devices propagate.** ``float32``/``float64`` inputs keep
  their precision end to end (integers promote to ``float32``), and CUDA
  inputs stay on CUDA — see :doc:`dtypes_devices`.

Where scikit-learn still wins
-----------------------------

- **Breadth**: dozens more estimators, solvers, and datasets.
- **Raw CPU speed** in hot loops (Cython/C extensions), e.g. tree building.
- **Battle-testing**: decades of edge-case coverage. torml is young, tested
  on CPU only, and documents its simplifications (binary-only
  ``LogisticRegression``/``LinearSVC``, single-target ``PLSRegression``,
  transductive ``LabelPropagation``).
