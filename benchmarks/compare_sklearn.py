"""Head-to-head correctness and timing: torml vs scikit-learn.

Same data, same seeds, CPU. Run with:
    uv run --with scikit-learn python benchmarks/compare_sklearn.py

scikit-learn is a benchmark-only dependency (never imported by torml).
"""

from __future__ import annotations

import time

import torch

ROW = "{0:<28} {1:<22} {2:<18} {3:<18}"


def timed(fn, repeats=3):
    """Return (best_seconds, last_result) over repeats."""
    best, out = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - start
        best = min(best, dt)
    return best, out


def main() -> None:
    """Run the comparison suite and print a table."""
    print(ROW.format("task", "metric", "torml", "sklearn"))
    print("-" * 88)

    # Linear regression -------------------------------------------------
    torch.manual_seed(0)
    X = torch.randn(5000, 20)
    true_w = torch.randn(20)
    y = X @ true_w + 0.1 * torch.randn(5000)
    Xn, yn = X.numpy(), y.numpy()

    from sklearn.linear_model import LinearRegression as SkLin
    from torml.linear_model import LinearRegression as ToLin

    tm, m = timed(lambda: ToLin().fit(X, y))
    ts, s = timed(lambda: SkLin().fit(Xn, yn))
    from torml.metrics import r2_score

    print(ROW.format("linreg fit", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    print(
        ROW.format(
            "linreg", "R2", f"{r2_score(y, m.predict(X)):.6f}", f"{s.score(Xn, yn):.6f}"
        )
    )

    # KMeans -------------------------------------------------------------
    torch.manual_seed(1)
    Xk = torch.randn(2000, 10)
    Xkn = Xk.numpy()

    from sklearn.cluster import KMeans as SkKM
    from torml.cluster import KMeans as ToKM

    tm, m = timed(lambda: ToKM(n_clusters=10, random_state=0, n_init=3).fit(Xk))
    ts, s = timed(
        lambda: SkKM(n_clusters=10, random_state=0, n_init=3).fit(Xkn)
    )
    print(ROW.format("kmeans fit", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    print(ROW.format("kmeans", "inertia", f"{m.inertia_:.1f}", f"{s.inertia_:.1f}"))

    # kNN classification ---------------------------------------------------
    torch.manual_seed(2)
    Xc = torch.randn(1000, 10)
    yc = (Xc[:, 0] > 0).long()
    Xcn, ycn = Xc.numpy(), yc.numpy()

    from sklearn.neighbors import KNeighborsClassifier as SkKNN
    from torml.neighbors import KNeighborsClassifier as ToKNN

    tm, m = timed(lambda: ToKNN(5).fit(Xc, yc))
    ts, s = timed(lambda: SkKNN(5).fit(Xcn, ycn))
    print(ROW.format("knn fit", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    tm, pm = timed(lambda: m.predict(Xc))
    ts, ps = timed(lambda: s.predict(Xcn))
    print(ROW.format("knn predict-1000", "seconds", f"{tm:.4f}", f"{ts:.4f}"))
    from torml.metrics import accuracy_score

    print(
        ROW.format(
            "knn",
            "accuracy",
            f"{accuracy_score(yc, pm):.4f}",
            f"{(ps == ycn).mean():.4f}",
        )
    )

    # Decision tree ---------------------------------------------------------
    from sklearn.tree import DecisionTreeClassifier as SkTree
    from torml.tree import DecisionTreeClassifier as ToTree

    tm, m = timed(lambda: ToTree(max_depth=6).fit(Xc, yc))
    ts, s = timed(lambda: SkTree(max_depth=6).fit(Xcn, ycn))
    print(ROW.format("tree fit", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    print(
        ROW.format(
            "tree",
            "accuracy",
            f"{accuracy_score(yc, m.predict(Xc)):.4f}",
            f"{s.score(Xcn, ycn):.4f}",
        )
    )

    # Scaling ----------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler as SkSS
    from torml.preprocessing import StandardScaler as ToSS

    tm, mt = timed(lambda: ToSS().fit_transform(X))
    ts, st = timed(lambda: SkSS().fit_transform(Xn))
    print(ROW.format("scaler", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    print(
        ROW.format(
            "scaler",
            "max |diff|",
            f"{float((mt - torch.as_tensor(st)).abs().max()):.2e}",
            "reference",
        )
    )

    # Autograd interop (torch-only) -------------------------------------------
    m2 = ToLin().fit(X, y)
    Xq = X[:4].detach().requires_grad_()
    m2.predict(Xq).sum().backward()
    print(
        ROW.format(
            "torch-native",
            "backward() thru predict",
            str(Xq.grad is not None),
            "n/a (numpy out)",
        )
    )


if __name__ == "__main__":
    main()
