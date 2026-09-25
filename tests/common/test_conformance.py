"""Conformance battery: check_estimator over naming-compliant estimators."""

from __future__ import annotations

import pytest
import torch

from torml.base import check_estimator as base_check
from torml.cluster import KMeans
from torml.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from torml.linear_model import LinearRegression, LogisticRegression
from torml.mixture import GaussianMixture
from torml.multivariate import MultiOutputRegressor
from torml.naive_bayes import GaussianNB
from torml.neighbors import KNeighborsClassifier, KNeighborsRegressor
from torml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torml.utils import check_estimator, check_estimator as utils_check


@pytest.fixture
def blobs():
    torch.manual_seed(0)
    x0 = torch.randn(24, 2) + torch.tensor([-3.0, 0.0])
    x1 = torch.randn(24, 2) + torch.tensor([3.0, 0.0])
    X = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(24), torch.ones(24)]).long()
    return X, y


@pytest.fixture
def regression_data():
    torch.manual_seed(1)
    X = torch.randn(30, 2)
    y = 2 * X[:, 0] - X[:, 1]
    return X, y


@pytest.mark.parametrize(
    "make",
    [
        lambda: LinearRegression(),
        lambda: LogisticRegression(),
        lambda: GaussianNB(),
        lambda: KMeans(n_clusters=2),
        lambda: GaussianMixture(n_components=2, random_state=0),
        lambda: DecisionTreeClassifier(),
        lambda: DecisionTreeRegressor(),
        lambda: KNeighborsClassifier(3),
        lambda: KNeighborsRegressor(3),
        lambda: BaggingClassifier(n_estimators=2, random_state=0),
        lambda: BaggingRegressor(n_estimators=2, random_state=0),
        lambda: RandomForestClassifier(n_estimators=2, random_state=0),
        lambda: RandomForestRegressor(n_estimators=2, random_state=0),
    ],
    ids=[
        "LinearRegression",
        "LogisticRegression",
        "GaussianNB",
        "KMeans",
        "GaussianMixture",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "KNeighborsClassifier",
        "KNeighborsRegressor",
        "BaggingClassifier",
        "BaggingRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
    ],
)
def test_check_estimator(make):
    """Test the conformance battery passes without raising."""
    assert check_estimator(make()) is not None


def test_both_import_paths_agree(blobs):
    """Test base and utils check_estimator are the same check."""
    X, y = blobs
    assert base_check is utils_check
    assert base_check(DecisionTreeClassifier()) is not None
    assert utils_check(KNeighborsClassifier(3)) is not None


def test_voting_and_multioutput_conform(blobs, regression_data):
    """Test meta-estimators with nested params conform."""
    X, y = blobs
    Xr, yr = regression_data
    assert (
        check_estimator(
            VotingClassifier(
                [("tree", DecisionTreeClassifier()), ("knn", KNeighborsClassifier(3))]
            )
        )
        is not None
    )
    assert (
        check_estimator(
            VotingRegressor(
                [("lin", LinearRegression()), ("tree", DecisionTreeRegressor())]
            )
        )
        is not None
    )
    assert check_estimator(MultiOutputRegressor(LinearRegression())) is not None
    _ = (X, y, Xr, yr)
