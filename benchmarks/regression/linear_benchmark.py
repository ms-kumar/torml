"""Micro-benchmark LinearRegression fit/predict.

Run with ``python benchmarks/regression/linear_benchmark.py``.
Excluded from the sdist (see MANIFEST.in) and from pytest collection.
"""

from __future__ import annotations

import time

import torch

from torml.linear_model import LinearRegression


def bench(n_samples: int = 5000, n_features: int = 20, repeats: int = 5) -> None:
    """Time LinearRegression fit/predict and print mean milliseconds."""
    torch.manual_seed(0)
    X = torch.randn(n_samples, n_features)
    y = X.sum(dim=1)

    fit_times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model = LinearRegression().fit(X, y)
        fit_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(X)
    predict_total = time.perf_counter() - start

    print(f"fit mean: {sum(fit_times) / len(fit_times) * 1000:.2f} ms")
    print(f"predict mean: {predict_total / repeats * 1000:.2f} ms")


if __name__ == "__main__":
    bench()
