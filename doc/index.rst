torml: PyTorch-backed scikit-learn-style ML library
====================================================

``torml`` reimplements the scikit-learn estimator API (``fit`` / ``predict`` /
``transform`` / ``score`` / ``get_params`` / ``set_params``) from scratch with
PyTorch (``torch.Tensor``, ``torch.linalg``) as the numerical backend —
no NumPy, SciPy, or scikit-learn at runtime.

Start with :doc:`quickstart`, learn the patterns in
:doc:`user_guide/supervised`, or jump straight to :doc:`api`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   user_guide/supervised
   user_guide/unsupervised
   user_guide/preprocessing
   user_guide/model_selection
   user_guide/dtypes_devices
   user_guide/comparison
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
