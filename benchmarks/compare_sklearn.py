"""Head-to-head correctness and timing: torml vs scikit-learn.

Same data, same seeds, CPU. Run with:
    uv run --with scikit-learn python benchmarks/compare_sklearn.py

scikit-learn is a benchmark-only dependency (never imported by torml).
"""

# pylint: disable=import-error

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


def bench_linreg():
    """Compare least-squares fits."""
    from sklearn.linear_model import LinearRegression as SkLin

    from torml.linear_model import LinearRegression as ToLin
    from torml.metrics import r2_score

    torch.manual_seed(0)
    features = torch.randn(10000, 20)
    target = features @ torch.randn(20) + 0.1 * torch.randn(10000)
    np_x, np_y = features.numpy(), target.numpy()
    tm, mine = timed(lambda: ToLin().fit(features, target))
    ts, theirs = timed(lambda: SkLin().fit(np_x, np_y))
    print(ROW.format("linreg fit", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    print(
        ROW.format(
            "linreg",
            "R2",
            f"{r2_score(target, mine.predict(features)):.6f}",
            f"{theirs.score(np_x, np_y):.6f}",
        )
    )
    return features, target


def bench_kmeans():
    """Compare k-means fits."""
    from sklearn.cluster import KMeans as SkKM

    from torml.cluster import KMeans as ToKM

    torch.manual_seed(1)
    data = torch.randn(10000, 10)
    tm, mine = timed(lambda: ToKM(n_clusters=10, random_state=0, n_init=3).fit(data))
    ts, theirs = timed(
        lambda: SkKM(n_clusters=10, random_state=0, n_init=3).fit(data.numpy())
    )
    print(ROW.format("kmeans fit", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    print(
        ROW.format(
            "kmeans", "inertia", f"{mine.inertia_:.1f}", f"{theirs.inertia_:.1f}"
        )
    )


def bench_knn():
    """Compare kNN fit/predict and accuracy."""
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    from torml.metrics import accuracy_score
    from torml.neighbors import KNeighborsClassifier as ToKNN

    torch.manual_seed(2)
    features = torch.randn(10000, 10)
    labels = (features[:, 0] > 0).long()
    np_x, np_y = features.numpy(), labels.numpy()
    tm, mine = timed(lambda: ToKNN(5).fit(features, labels))
    ts, theirs = timed(lambda: SkKNN(5).fit(np_x, np_y))
    print(ROW.format("knn fit", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    tm, my_pred = timed(lambda: mine.predict(features))
    ts, their_pred = timed(lambda: theirs.predict(np_x))
    print(ROW.format("knn predict-10k", "seconds", f"{tm:.4f}", f"{ts:.4f}"))
    print(
        ROW.format(
            "knn",
            "accuracy",
            f"{accuracy_score(labels, my_pred):.4f}",
            f"{(their_pred == np_y).mean():.4f}",
        )
    )
    return features, labels


def bench_tree(features, labels):
    """Compare decision tree fits on the kNN data."""
    from sklearn.tree import DecisionTreeClassifier as SkTree

    from torml.metrics import accuracy_score
    from torml.tree import DecisionTreeClassifier as ToTree

    np_x, np_y = features.numpy(), labels.numpy()
    tm, mine = timed(lambda: ToTree(max_depth=6).fit(features, labels))
    ts, theirs = timed(lambda: SkTree(max_depth=6).fit(np_x, np_y))
    print(ROW.format("tree fit", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    print(
        ROW.format(
            "tree",
            "accuracy",
            f"{accuracy_score(labels, mine.predict(features)):.4f}",
            f"{theirs.score(np_x, np_y):.4f}",
        )
    )


def bench_scaler():
    """Compare scalers and their numerical agreement."""
    from sklearn.preprocessing import StandardScaler as SkSS

    from torml.preprocessing import StandardScaler as ToSS

    torch.manual_seed(0)
    data = torch.randn(10000, 20)
    tm, mine = timed(lambda: ToSS().fit_transform(data))
    ts, theirs = timed(lambda: SkSS().fit_transform(data.numpy()))
    print(ROW.format("scaler", "seconds (best of 3)", f"{tm:.4f}", f"{ts:.4f}"))
    print(
        ROW.format(
            "scaler",
            "max |diff|",
            f"{float((mine - torch.as_tensor(theirs)).abs().max()):.2e}",
            "reference",
        )
    )


def bench_autograd():
    """Show predictions stay differentiable (torch-only)."""
    from torml.linear_model import LinearRegression as ToLin

    torch.manual_seed(0)
    features = torch.randn(50, 3)
    target = features @ torch.tensor([1.0, 2.0, -1.0])
    model = ToLin().fit(features, target)
    query = features[:4].detach().requires_grad_()
    model.predict(query).sum().backward()
    print(
        ROW.format(
            "torch-native",
            "backward() thru predict",
            str(query.grad is not None),
            "n/a (numpy out)",
        )
    )


def main() -> None:
    """Run the comparison suite and print a table."""
    print(ROW.format("task", "metric", "torml", "sklearn"))
    print("-" * 88)
    bench_linreg()
    bench_kmeans()
    features, labels = bench_knn()
    bench_tree(features, labels)
    bench_scaler()
    bench_autograd()


if __name__ == "__main__":
    main()
