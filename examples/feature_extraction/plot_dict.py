"""Vectorize dict rows with DictVectorizer.

Run with ``python examples/feature_extraction/plot_dict.py``.
"""

from __future__ import annotations

from torml.feature_extraction import DictVectorizer


def main() -> None:
    """Run the dict vectorizer example."""
    rows = [{"city": 1, "temp": 21.5}, {"city": 2, "humidity": 0.8}]
    vec = DictVectorizer().fit(rows)
    print(f"names: {vec.get_feature_names_out()}")
    print(f"matrix: {vec.transform(rows).tolist()}")


if __name__ == "__main__":
    main()
