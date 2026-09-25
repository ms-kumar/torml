torml vs scikit-learn
======================

torml mirrors the scikit-learn API on a PyTorch backend. This page reports a
fair, reproducible head-to-head so you know exactly where each library wins.
Rerun it any time:

.. code-block:: bash

   uv run --with scikit-learn python benchmarks/compare_sklearn.py

Method: same data, same seeds, CPU, best of 3 timings
(Apple M-series, 2026-09-25, torch 2.14.0, scikit-learn 1.9.1).

Results
-------

=================== ======================= ================== ==================
task                metric                  torml              scikit-learn
=================== ======================= ================== ==================
linreg fit          seconds (best of 3)     0.0006             0.0009
linreg              R²                      0.999596           0.999596
kmeans fit          seconds (best of 3)     0.0292             0.0278
kmeans              inertia (lower better)  14235.3            14271.8
knn fit             seconds (best of 3)     0.0000             0.0003
knn predict-1000    seconds                 0.0055             0.0057
knn                 accuracy                0.9240             0.9240
tree fit            seconds (best of 3)     0.0010             0.0006
tree                accuracy                1.0000             1.0000
scaler              seconds (best of 3)     0.0005             0.0005
scaler              max |diff| vs sklearn   4.77e-07           reference
torch-native        backward() thru predict True               n/a (numpy out)
=================== ======================= ================== ==================

Reading the table honestly: correctness matches everywhere (identical R²,
accuracy, and scaler outputs; kmeans inertia differs only by random
initialization). Timings are in the same class except tree fitting, where
scikit-learn's Cython is ~2x faster than torml's Python CART loops.

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

Where scikit-learn still wins
-----------------------------

- **Breadth**: dozens more estimators, solvers, and datasets.
- **Raw CPU speed** in hot loops (Cython/C extensions), e.g. tree building.
- **Battle-testing**: decades of edge-case coverage. torml is young, tested
  on CPU only, and documents its simplifications (binary-only
  ``LogisticRegression``/``LinearSVC``, single-target ``PLSRegression``,
  transductive ``LabelPropagation``).
