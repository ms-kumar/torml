"""torml (CPU vs MPS) vs scikit-learn (CPU) at ~50k samples.

Covers what fits in memory honestly: kNN predicts 2k queries (a full
all tasks run at a uniform 50k samples; kNN predicts 2k queries.

Run with: uv run --with scikit-learn python benchmarks/compare_500k.py
Needs an MPS device for the torml-MPS column, else it is skipped.
"""

# pylint: disable=import-error

from __future__ import annotations

import time

import torch

ROW = "{0:<16} {1:<24} {2:<14} {3:<14} {4:<14}"

MPS = torch.backends.mps.is_available()


def timed(fn, repeats=3):
    """Return (best_seconds, last_result); first call warms up MPS kernels."""
    best, out = float("inf"), None
    for _ in range(repeats + 1):
        start = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - start
        if _ > 0:
            best = min(best, dt)
    return best, out


def fmt(seconds):
    """Format seconds, '-' when skipped."""
    return f"{seconds:.3f}" if seconds is not None else "-"


def bench_linreg():
    """Least squares, 50k x 20."""
    from sklearn.linear_model import LinearRegression as SkLin

    from torml.linear_model import LinearRegression as ToLin
    from torml.metrics import r2_score

    torch.manual_seed(0)
    cpu_x = torch.randn(50000, 20)
    cpu_y = cpu_x @ torch.randn(20) + 0.1 * torch.randn(50000)
    tc, mc = timed(lambda: ToLin().fit(cpu_x, cpu_y))
    ts, ss = timed(lambda: SkLin().fit(cpu_x.numpy(), cpu_y.numpy()))
    tm = None
    if MPS:
        mps_x, mps_y = cpu_x.to("mps"), cpu_y.to("mps")
        tm, mm = timed(lambda: ToLin().fit(mps_x, mps_y))
        r2m = float(r2_score(mps_y, mm.predict(mps_x)))
    else:
        r2m = float("nan")
    print(ROW.format("linreg fit", "s best-of-3", fmt(tc), fmt(tm), fmt(ts)))
    print(
        ROW.format(
            "linreg",
            "R2",
            f"{r2_score(cpu_y, mc.predict(cpu_x)):.6f}",
            f"{r2m:.6f}" if MPS else "-",
            f"{ss.score(cpu_x.numpy(), cpu_y.numpy()):.6f}",
        )
    )


def bench_scaler():
    """Standard scaling, 50k x 20."""
    from sklearn.preprocessing import StandardScaler as SkSS

    from torml.preprocessing import StandardScaler as ToSS

    torch.manual_seed(0)
    cpu_x = torch.randn(50000, 20)
    tc, mc = timed(lambda: ToSS().fit_transform(cpu_x))
    ts, _ = timed(lambda: SkSS().fit_transform(cpu_x.numpy()))
    tm = None
    if MPS:
        mps_x = cpu_x.to("mps")
        tm, mm = timed(lambda: ToSS().fit_transform(mps_x))
        diff = float((mc - mm.cpu()).abs().max())  # float32: MPS lacks float64
    else:
        diff = float("nan")
    print(ROW.format("scaler", "s best-of-3", fmt(tc), fmt(tm), fmt(ts)))
    print(ROW.format("scaler", "cpu-vs-mps maxdiff", f"{diff:.2e}", "-", "-"))


def bench_kmeans():
    """KMeans k=10, 50k x 10, 2 inits."""
    from sklearn.cluster import KMeans as SkKM

    from torml.cluster import KMeans as ToKM

    torch.manual_seed(1)
    cpu_x = torch.randn(50000, 10)
    tc, mc = timed(lambda: ToKM(n_clusters=10, random_state=0, n_init=2).fit(cpu_x))
    ts, ss = timed(
        lambda: SkKM(n_clusters=10, random_state=0, n_init=2).fit(cpu_x.numpy())
    )
    tm = None
    if MPS:
        mps_x = cpu_x.to("mps")
        tm, mm = timed(
            lambda: ToKM(n_clusters=10, random_state=0, n_init=2).fit(mps_x)
        )
        inert_m = mm.inertia_
    else:
        inert_m = float("nan")
    print(ROW.format("kmeans fit", "s best-of-3", fmt(tc), fmt(tm), fmt(ts)))
    print(
        ROW.format(
            "kmeans", "inertia", f"{mc.inertia_:.1f}", f"{inert_m:.1f}", f"{ss.inertia_:.1f}"
        )
    )


def bench_knn():
    """kNN fit on 50k, predict 2k queries."""
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    from torml.metrics import accuracy_score
    from torml.neighbors import KNeighborsClassifier as ToKNN

    torch.manual_seed(2)
    cpu_x = torch.randn(50000, 10)
    cpu_y = (cpu_x[:, 0] > 0).long()
    query = torch.randn(2000, 10)
    query_y = (query[:, 0] > 0).long()
    tc, mc = timed(lambda: ToKNN(5).fit(cpu_x, cpu_y))
    ts, ss = timed(lambda: SkKNN(5).fit(cpu_x.numpy(), cpu_y.numpy()))
    print(ROW.format("knn fit-50k", "s best-of-3", fmt(tc), "-", fmt(ts)))
    tc, pc = timed(lambda: mc.predict(query))
    ts, ps = timed(lambda: ss.predict(query.numpy()))
    row_sk = f"{(ps == query_y.numpy()).mean():.4f}"
    row_mps, tm = "-", None
    if MPS:
        mps_x, mps_y, mps_q = cpu_x.to("mps"), cpu_y.to("mps"), query.to("mps")
        mm = ToKNN(5).fit(mps_x, mps_y)
        tm, pm = timed(lambda: mm.predict(mps_q))
        row_mps = f"{accuracy_score(query_y, pm.cpu()):.4f}"
    print(ROW.format("knn predict-2k", "seconds", fmt(tc), fmt(tm), fmt(ts)))
    print(
        ROW.format(
            "knn", "accuracy", f"{accuracy_score(query_y, pc):.4f}", row_mps, row_sk
        )
    )


def bench_tree():
    """Decision tree depth 6 at 50k."""
    from sklearn.tree import DecisionTreeClassifier as SkTree

    from torml.metrics import accuracy_score
    from torml.tree import DecisionTreeClassifier as ToTree

    torch.manual_seed(3)
    cpu_x = torch.randn(50000, 10)
    cpu_y = (cpu_x[:, 0] > 0).long()
    tc, mc = timed(lambda: ToTree(max_depth=6).fit(cpu_x, cpu_y))
    ts, ss = timed(lambda: SkTree(max_depth=6).fit(cpu_x.numpy(), cpu_y.numpy()))
    row_mps, tm = "-", None
    acc_mps = "-"
    if MPS:
        mps_x, mps_y = cpu_x.to("mps"), cpu_y.to("mps")
        tm, mm = timed(lambda: ToTree(max_depth=6).fit(mps_x, mps_y))
        acc_mps = f"{accuracy_score(mps_y, mm.predict(mps_x)):.4f}"
    print(ROW.format("tree fit-50k", "s best-of-3", fmt(tc), fmt(tm), fmt(ts)))
    print(
        ROW.format(
            "tree",
            "accuracy",
            f"{accuracy_score(cpu_y, mc.predict(cpu_x)):.4f}",
            acc_mps,
            f"{ss.score(cpu_x.numpy(), cpu_y.numpy()):.4f}",
        )
    )


def main() -> None:
    """Run the 50k comparison and print a table."""
    print(ROW.format("task", "metric", "torml-cpu", "torml-mps", "sklearn-cpu"))
    print("-" * 88)
    if not MPS:
        print("(no MPS device: torml-mps column skipped)")
    bench_linreg()
    bench_scaler()
    bench_kmeans()
    bench_knn()
    bench_tree()


if __name__ == "__main__":
    main()
