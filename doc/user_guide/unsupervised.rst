Unsupervised learning
=====================

Learn structure from ``X`` alone — no ``y``. These estimators implement
``fit``; clusterers additionally implement ``fit_predict``.

Clustering
----------

- ``KMeans``: Lloyd iterations from random restarts; ``labels_`` holds the
  closest-center index, ``cluster_centers_`` the centers, ``inertia_`` the
  compactness. Run several ``n_init`` restarts (best inertia wins) and pick
  ``n_clusters`` by the elbow in ``inertia_``.
- ``DBSCAN``: groups density-connected points; labels ``-1`` mark noise.
  Good when clusters have odd shapes or noise must be isolated — tune
  ``eps`` (neighborhood radius) and ``min_samples`` together.
- ``GaussianMixture``: soft assignments via EM over full covariances;
  ``predict_proba`` gives responsibilities and ``score`` the mean log
  likelihood. Use it when clusters overlap and hard labels mislead.

.. code-block:: python

   from torml.cluster import KMeans

   km = KMeans(n_clusters=3, random_state=0).fit(X)
   km.labels_, km.inertia_

Dimensionality reduction and manifolds
--------------------------------------

- ``PCA``: full-SVD projection keeping maximum variance;
  ``inverse_transform`` reconstructs approximately. ``n_components`` accepts
  a count or a variance fraction (e.g. ``0.95``).
- ``MDS``: classical scaling that preserves pairwise distances; accepts a
  ``dissimilarity="precomputed"`` distance matrix.

Density and covariance
----------------------

- ``EmpiricalCovariance``: mean, covariance, and precision plus
  ``mahalanobis`` distances and a Gaussian ``score`` — the standard tool
  for spotting outliers far from the bulk of the data.

Tips
----

- Scale features before distance-based methods (kNN graphs, MDS, DBSCAN):
  raw units distort neighborhoods. See :doc:`preprocessing`.
